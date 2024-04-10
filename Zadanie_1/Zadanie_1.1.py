import geopandas as gpd
from owslib.wfs import WebFeatureService
import requests
from typing import Union
from pathlib import Path
import xml.etree.ElementTree as ET
from osgeo import gdal
import zipfile
import os
import shapely
import rasterio
from rasterio.mask import mask
import open3d
import numpy as np
import shapely.geometry
import argparse

def get_geom(file_path: str, id: int) -> tuple:
    if file_path == 'hextiles 1.fgb':
        data = gpd.read_file(file_path)
        geom = data['geometry'][id]
        
        bbox = geom.bounds
        geom = geom.exterior.coords
        geom = list(geom[:-1])
        return geom, bbox

def get_nmt_nmpt(service: str , geom_bbox, folder) -> list:
        #downloads NMT or NMPT tiles for given geometry
        #converts downloaded asc files to tif files
        #removes asc files after conversion
    
    if service == 'NMT':
        url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NumerycznyModelTerenuEVRF2007/WFS/Skorowidze"
    elif service == 'NMPT':
        url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NumerycznyModelPokryciaTerenuEVRF2007/WFS/Skorowidze"
    else:
        raise ValueError("Invalid service name: only 'NMT' and 'NMPT' are allowed.")
    
    wfs_service = WebFeatureService(url=url, version='2.0.0')
    content = list(wfs_service.contents)
    
    response = wfs_service.getfeature(typename=content[-1], bbox=geom_bbox, propertyname=['gugik:url_do_pobrania', 'char_przestrz'])
    tree = ET.parse(response)
    root = tree.getroot()

    elements = root.findall('.//*[{http://www.gugik.gov.pl}url_do_pobrania][{http://www.gugik.gov.pl}char_przestrz]')

    urls = []
    for element in elements:
        url_element = element.find('{http://www.gugik.gov.pl}url_do_pobrania')
        char_przestrz_element = element.find('{http://www.gugik.gov.pl}char_przestrz')
        
        url = url_element.text
        char_przestrz = char_przestrz_element.text
        
        if service == 'NMT' and char_przestrz == '1.00 m':
            urls.append(url)
        elif service == 'NMPT':
            urls.append(url)
    
    tifs = []
    for i in range(len(urls)):
        asc_file = f"{service}_file{i}.asc"
        download_and_save_file(urls[i], asc_file, folder)
        gdal.Translate(f"{folder}/{service}_file{i}.tif", f"{folder}/{service}_file{i}.asc")
        os.remove(f"{folder}/{service}_file{i}.asc")
        tifs.append(f"{folder}/{service}_file{i}.tif")
        
    return tifs

def extract_file(zip_file: str, file_to_extract: str, folder: str) -> str:
        #extracts files containing 'file_to_extract' string from given zip file
        #deletes zip file after extraction
    
    file_name = None
    zip_path = os.path.join(folder, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file_to_extract in file:
                file_name = file
                zip_ref.extract(file, folder)
                extracted_path = os.path.join(folder, file)
                os.rename(extracted_path, os.path.join(folder, os.path.basename(file)))
    os.remove(zip_path)
    
    return os.path.basename(file_name)

def get_bdot(geom_bbox, folder) -> str:
        #downloads BDOT10k zip files for given geometry
        #extracts BUBD_A files from downloaded zip files
    
    url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/BDOT/WFS/PobieranieBDOT10k"
    service = WebFeatureService(url=url, version='2.0.0')
    content = list(service.contents)
    response = service.getfeature(typename=content[-1], bbox=geom_bbox, propertyname=['ms:URL_GML'])
    tree = ET.parse(response)
    root = tree.getroot()
    elements = root.findall('.//ms:URL_GML', {'ms':'http://mapserver.gis.umn.edu/mapserver'})
    
    urls = []
    for element in elements:
        text = element.text
        urls.append(text)
    
    zips = []
    for i in range(len(urls)):
        zip_file = f"file{i}.zip"
        download_and_save_file(urls[i], zip_file, folder)
        zips.append(zip_file)
    
    for i in range(len(zips)):
        file_name = extract_file(zips[i], 'BUBD_A', folder)
    
    return file_name
        
def merge_tiles(sections: list, folder: str, merged_name: str) -> None:
        #merges tif tiles into one and removes individual ones
    
    tile = gdal.Warp(f"{folder}/{merged_name}.tif", sections, format="GTiff")
    tile = None
    for section in sections:
        os.remove(section)

def download_and_save_file(download_url: str, save_path: Union[Path, str], folder: str) -> None:
        #saves downloaded file to given folder
        #creates folder if doesn't exist
    
    response = requests.get(download_url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download {download_url}. Status code: {response.status_code}")
    folder = Path(f"{folder}")
    folder.mkdir(parents=True, exist_ok=True)
    
    with open(folder / save_path, "wb") as file:
        file.write(response.content)

def clip_raster(bounds, raster: str, clipped_file_name: str, folder: str = None) -> None:
        #clips raster to given bounds
        #saves clipped raster to given folder, deletes original one
    
    aoi = shapely.Polygon(bounds)
    raster_path = f"{folder}/{raster}"
    with rasterio.open(raster_path, 'r') as src:
        out_image, out_transform = mask(src, [aoi], crop=True)
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    output_file = f"{folder}/{clipped_file_name}.tif"
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(out_image)
    os.remove(raster_path)
    
def clip_vector_xml(bounds, vector: str, clipped_file_name: str, folder: str = None) -> None:
        #clips vector from given xml file
        #checks for invalid data types and converts them to string
    
    aoi = shapely.Polygon(bounds)
    buildings = gpd.read_file(f"{folder}/{vector}")
    buildings_clip = buildings[buildings.within(aoi)]
    output_path = f"{folder}/{clipped_file_name}.xml"
    for col in buildings_clip.columns:
        if buildings_clip[col].dtype == 'object':
            buildings_clip.loc[:,col] = buildings_clip[col].astype(str)
    buildings_clip.to_file(output_path)
    
def read_raster(filename: str):
        #reads raster with gdal
        #changes nodata values to 0
    
    raster = gdal.Open(filename, gdal.GA_Update)
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    b_array = band.ReadAsArray()
    b_array[b_array == -9999] = 0
    band.WriteArray(b_array)
    
    return raster    

def create_vertex_array(raster):
    transform = raster.GetGeoTransform()
    width = raster.RasterXSize
    height = raster.RasterYSize
    x = np.arange(0, width) * transform[1] + transform[0]
    y = np.arange(0, height) * transform[5] + transform[3]
    xx, yy = np.meshgrid(x, y)
    zz = raster.ReadAsArray()
    vertices = np.vstack((xx, yy, zz)).reshape([3, -1]).transpose()
    
    return vertices

def create_index_array(raster):
    width = raster.RasterXSize
    height = raster.RasterYSize

    ai = np.arange(0, width - 1)
    aj = np.arange(0, height - 1)
    aii, ajj = np.meshgrid(ai, aj)
    a = aii + ajj * width
    a = a.flatten()

    tria = np.vstack((a, a + width, a + width + 1, a, a + width + 1, a + 1))
    tria = np.transpose(tria).reshape([-1, 3])
    
    return tria

def polygon_to_3d(polygon, max_height: float):
        #from given 2D polygon creates 3D mesh
    
    min_height = 0.0

    xy = polygon.exterior.coords
    xy = list(xy[:-1])
    xy = np.float64(xy)

    xyz = np.hstack([xy, np.full((len(xy), 1), min_height)])
    xyh = np.hstack([xy, np.full((len(xy), 1), max_height)])

    xyz = np.vstack([xyz, xyh])

    num_vertices = len(xy)
    triangles = []
    
        #creates triangles for the sides of the cube
    for i in range(len(xy)):
            i_new = (i+1) % len(xy)
            triangles.append([i, i_new, i + num_vertices])
            triangles.append([i_new, i_new + num_vertices, i + num_vertices])
    
    bottom_face = list(range(num_vertices))
    top_face = list(range(num_vertices, 2 * num_vertices))

    for i in range(num_vertices - 2):
        triangles.append([bottom_face[0], bottom_face[i + 1], bottom_face[i + 2]])

    for i in range(num_vertices - 2):
        triangles.append([top_face[0], top_face[i + 1], top_face[i + 2]])

    triangles.append([bottom_face[0], bottom_face[-1], top_face[0]])
    triangles.append([bottom_face[-1], top_face[-1], top_face[0]])

    mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(xyz),
        open3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    
    return mesh

def height(polygon, raster_file: str, folder: str) -> float:
        #creates mask on raster with polygon
        #removes nodata values(-9999)
        #from the rest of the data calculates avarage
    
    with rasterio.open(f"{folder}/{raster_file}") as src:
        geom = shapely.geometry.mapping(polygon)
        masked_data, _ = mask(src, [geom], crop=True)
        height_values = masked_data[masked_data != -9999]
        if len(height_values) > 0:
            avg_height = height_values.mean()
        else:
            avg_height = 0
        
    return avg_height

def parse_args():
        #arguments from command line
    
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to the file with hexagons' geometries")
    parser.add_argument("idx", type=int, help="Index of the chosen hexagon")
    parser.add_argument("folder", type=str, help="Folder to save downloaded files")
    parser.add_argument("--mesh_model", type=str, help="Path for saving the mesh model [.ply]")
    mesh_model = None
    args = parser.parse_args()
    if args.mesh_model:
        mesh_model = args.mesh_model
        
    return args.file_path, args.idx, args.folder, mesh_model
    
def main():
    file_path, idx, folder, mesh_model = parse_args()
    
        #downloading and merging necessary data
    geom, bbox = get_geom(f'{file_path}', idx)
    
    nmt_tifs = get_nmt_nmpt('NMT', bbox, folder)
    merge_tiles(nmt_tifs, folder, 'nmt_merged')
    
    nmpt = get_nmt_nmpt('NMPT', bbox, folder)
    merge_tiles(nmpt, folder, 'nmpt_merged')

    bdot_file = get_bdot(bbox, folder)
    
    
    #     #clipping data
    clip_raster(geom, 'nmt_merged.tif', 'clipped_nmt', folder)
    clip_raster(geom, 'nmpt_merged.tif', 'clipped_nmpt', folder)
    clip_vector_xml(geom, bdot_file, 'clipped_bubd_a', folder)
    
    
    #     #generating meshes
    raster = read_raster(f"{folder}/clipped_nmt.tif")
    vertices = create_vertex_array(raster)
    triangles = create_index_array(raster)
    
    mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(vertices),
        open3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    
    buildings = gpd.read_file(f"{folder}/clipped_bubd_a.xml")
    shapes = [x for x in buildings["geometry"]]
    
    for shape in shapes:
        base = height(shape, 'clipped_nmt.tif', folder)
        building_height = height(shape, 'clipped_nmpt.tif', folder)
        b_mesh = polygon_to_3d(shape, building_height-base)
        b_mesh.translate([0, 0, base])
        mesh += b_mesh
    
    mesh.translate(-mesh.get_center())
    
    open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    
    if mesh_model is not None:
        print(f"Saving mesh model to {mesh_model}")
        open3d.io.write_triangle_mesh("mesh_lod1.ply", mesh)
    
    

if __name__ == '__main__':
    main()