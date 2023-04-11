from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
import time
import numpy as np

# create an instance of the API class
apiClient = swagger_client.ApiClient()
api_instance = swagger_client.AssemblyServiceApi(swagger_client.ApiClient())
holding_service = swagger_client.RepositoryHoldingsServiceApi(apiClient)
entry_service = swagger_client.EntryServiceApi(apiClient)
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
assembly_id = 'assembly_id_example' # str | ASSEMBLY ID of the biological assembly candidate.

group_ids = set()
try:
    all_elements = holding_service.get_current_entry_ids()
    for idx, element in enumerate(all_elements):
        try:
            entry = entry_service.get_entry_by_id(element)
            group_ids.add(entry.rcsb_entry_container_identifiers.rcsb_id)
        except Exception as e:
            print("Has no group_id or request failed")
            print("Error: ", e)
        if idx % 1000 == 0:
            print("Processed: ", idx)
    group_ids = np.array(list(group_ids))
    np.savetxt('test.csv', group_ids, delimiter=',') 
except ApiException as e:
    print("Exception when calling AssemblyServiceApi->get_assembly_by_id: %s\n" % e)