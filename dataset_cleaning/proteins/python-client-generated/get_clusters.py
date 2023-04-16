from __future__ import print_function
import time
import swagger_client
from tqdm import tqdm

from swagger_client.rest import ApiException
import time
import numpy as np
import sys
import json

# create an instance of the API class
apiClient = swagger_client.ApiClient()
api_instance = swagger_client.AssemblyServiceApi(swagger_client.ApiClient())
group_service = swagger_client.GroupsServiceApi(apiClient)
holding_service = swagger_client.RepositoryHoldingsServiceApi(apiClient)
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
assembly_id = 'assembly_id_example' # str | ASSEMBLY ID of the biological assembly candidate.

print("Starting", file=sys.stdout)
def remove_numeric_suffix(input_string):
    # Split the string at the underscore
    parts = input_string.rsplit('_', 1)

    # Check if the last part consists of numbers
    if len(parts) > 1 and parts[-1].isdigit():
        # Remove the last part (including the underscore)
        result = parts[0]
        suffix = parts[1]
    else:
        result = input_string
        suffix = None
    return [result, suffix]

# Test the function with different input strings
#input_strings = ["1A00_1", "4WJG", "AF_AFP69906F1_1", "example_text_123", "example_text_not_numbers"]

try:
    #all_elements = holding_service.get_current_entry_ids()
    all_groups ={}
    all_cluster_ids = np.genfromtxt('cluster_ids.csv', delimiter='\n', dtype=str)
    print(all_cluster_ids)
    #all_cluster_ids = ["110_100"]
    for cluster_id in tqdm(all_cluster_ids):
        def get_member_id(element):
            return remove_numeric_suffix(element)#.split(r"_")[0]
        try:
            ids = group_service.get_polymer_entity_group_by_id(cluster_id).rcsb_group_container_identifiers.group_member_ids
            if len(ids) <= 1:
                continue
            group = list(map(get_member_id, ids))
            all_groups[cluster_id]=group
        except Exception as e:
            print("Has no group_id or request failed", file=sys.stdout)
            print(f"Error: {e}", file=sys.stdout)
    #np.array(all_groups)
    with open("all_clusters.json", "w") as f:
        json.dump(all_groups, f)
except ApiException as e:
    print("Exception when calling AssemblyServiceApi->get_assembly_by_id: %s\n" % e)