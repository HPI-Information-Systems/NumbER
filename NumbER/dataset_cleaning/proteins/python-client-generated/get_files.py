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
group_service = swagger_client.GroupsServiceApi(apiClient)
holding_service = swagger_client.RepositoryHoldingsServiceApi(apiClient)
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
assembly_id = 'assembly_id_example' # str | ASSEMBLY ID of the biological assembly candidate.

print("Starting", file=sys.stdout)
try:
    all_elements = np.array(holding_service.get_current_entry_ids())
    all_elements.tofile('output.csv', sep=',', format='%s')

    #np.savetxt('all_ids.csv', all_elements, delimiter=',',fmt='%s') 
except ApiException as e:
    print("Exception when calling AssemblyServiceApi->get_assembly_by_id: %s\n" % e)