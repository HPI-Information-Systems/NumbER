from __future__ import print_function
import time
import swagger_client
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
    else:
        result = input_string
    return result

# Test the function with different input strings
#input_strings = ["1A00_1", "4WJG", "AF_AFP69906F1_1", "example_text_123", "example_text_not_numbers"]

try:
    #all_elements = holding_service.get_current_entry_ids()
    all_groups =[]
    all_cluster_ids = np.genfromtxt('my_file.csv', delimiter=',')
    for cluster_id in all_cluster_ids:
        def get_member_id(element):
            return remove_numeric_suffix(element)#.split(r"_")[0]
        group = list(map(get_member_id, group_service.get_polymer_entity_group_by_id("110_100").rcsb_group_container_identifiers.group_member_ids))
        all_groups.append(group)
    #np.array(all_groups)
    with open("all_clusters.json", "w") as f:
        json.dump(all_groups, f)
except ApiException as e:
    print("Exception when calling AssemblyServiceApi->get_assembly_by_id: %s\n" % e)