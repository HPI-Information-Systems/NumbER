import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup as bs

with open("earthquakes.xml", "r") as file:
    content = file.readlines()
content = "".join(content)
bs_content = bs(content, "xml")
event_data = []
for event in bs_content.find_all("event"):
    event_id = event['catalog:eventid']
    origins = event.find_all("origin")
    temp = {}
    for origin in origins:
        if origin.get("catalog:datasource") != None:
            datasource = origin["catalog:datasource"]
        else:
            continue
        time = origin.time.value.text if origin.time and origin.time.value else None
        longitude = origin.longitude.value.text if origin.longitude and origin.longitude.value else None
        latitude = origin.latitude.value.text if origin.latitude and origin.latitude.value else None
        depth = origin.depth.value.text if origin.depth and origin.depth.value else None
        depth_uncertainty = origin.depth.uncertainty.text if origin.depth and origin.depth.uncertainty else None
        horizontal_uncertainty = origin.originUncertainty.horizontalUncertainty.text if origin.originUncertainty and origin.originUncertainty.horizontalUncertainty else None
        used_phase_count = origin.quality.usedPhaseCount.text if origin.quality and origin.quality.usedPhaseCount else None
        used_station_count = origin.quality.usedStationCount.text if origin.quality and origin.quality.usedStationCount  else None
        standard_error = origin.quality.standardError.text if  origin.quality and origin.quality.standardError else None
        azimuthal_gap = origin.quality.azimuthalGap.text if origin.quality and origin.quality.azimuthalGap else None
        minimum_distance = origin.quality.minimumDistance.text if origin.quality and origin.quality.minimumDistance else None
        temp[datasource] = {"time": time, "longitude": longitude, "latitude": latitude, "depth": depth, "depth_uncertainty": depth_uncertainty, "horizontal_uncertainty": horizontal_uncertainty, "used_phase_count": used_phase_count, "used_station_count": used_station_count, "standard_error": standard_error, "azimuthal_gap": azimuthal_gap, "minimum_distance": minimum_distance}
    #print(temp)
    for magnitude in event.find_all("magnitude"):
        if magnitude.get("catalog:datasource") != None:
            datasource = magnitude["catalog:datasource"]
        else:
            continue
        mag_value = magnitude.mag.value.text if magnitude.mag and magnitude.mag.value else None
        mag_uncertainty = magnitude.mag.uncertainty.text if magnitude.mag and magnitude.mag.uncertainty else None
        mag_type = magnitude.mag.type.text if magnitude.mag and magnitude.mag.type else None
        mag_station_count = magnitude.mag.stationCount.text if magnitude.mag and magnitude.mag.stationCount else None
        mag_creation_time = magnitude.creationInfo.creationTime.text if magnitude.creationInfo and magnitude.creationInfo.creationTime else None
        temp[datasource]["mag_value"] = mag_value
        temp[datasource]["mag_uncertainty"] = mag_uncertainty
        temp[datasource]["mag_type"] = mag_type
        temp[datasource]["mag_station_count"] = mag_station_count
        temp[datasource]["mag_creation_time"] = mag_creation_time
    for key, item in temp.items():
        item["event_id"] = event_id
        item["datasource"] = key
        event_data.append(item)
df = pd.DataFrame(event_data)
df.to_csv('./temp.csv', index=False)
        