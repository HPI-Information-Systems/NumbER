from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
import time
import numpy as np
import sys

# create an instance of the API class
apiClient = swagger_client.ApiClient()
api_instance = swagger_client.AssemblyServiceApi(swagger_client.ApiClient())
holding_service = swagger_client.RepositoryHoldingsServiceApi(apiClient)
entry_service = swagger_client.EntryServiceApi(apiClient)
entity_service = swagger_client.EntityServiceApi(apiClient)
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
assembly_id = 'assembly_id_example' # str | ASSEMBLY ID of the biological assembly candidate.

group_ids = set()
print("Starting", file=sys.stdout)
try:
    all_elements = holding_service.get_current_entry_ids()
    print("Got all elements", file=sys.stdout)
    for idx, element in enumerate(all_elements):
        element = entry_service.get_entry_by_id(element)
        entry_id = element.entry.id
        #print("entry_id", entry_id, file=sys.stdout)
        if element.rcsb_entry_container_identifiers.polymer_entity_ids != None:
            for entity_id in element.rcsb_entry_container_identifiers.polymer_entity_ids:
                #print("entity_id", entity_id, file=sys.stdout)
                try:
                    clusters = entity_service.get_polymer_entity_by_id(entry_id, entity_id).rcsb_cluster_membership
                    group_ids.update([f"{cluster.cluster_id}_100" for cluster in clusters if cluster.identity == 100])
                    #print(group_ids)
                except Exception as e:
                    print("Has no group_id or request failed", file=sys.stdout)
                    print(f"Error: {e}", file=sys.stdout)
            
                    #group_ids = np.array(list(group_ids))
    group_ids = np.array(list(group_ids))
    np.savetxt('cluster_ids.csv', group_ids, delimiter=',',fmt='%s') 
except ApiException as e:
    print("Exception when calling AssemblyServiceApi->get_assembly_by_id: %s\n" % e)