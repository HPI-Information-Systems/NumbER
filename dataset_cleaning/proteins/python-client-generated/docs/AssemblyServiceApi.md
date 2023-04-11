# swagger_client.AssemblyServiceApi

All URIs are relative to */rest/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_assembly_by_id**](AssemblyServiceApi.md#get_assembly_by_id) | **GET** /core/assembly/{entry_id}/{assembly_id} | Get structural assembly description by ENTRY ID and ASSEMBLY ID.

# **get_assembly_by_id**
> CoreAssembly get_assembly_by_id(entry_id, assembly_id)

Get structural assembly description by ENTRY ID and ASSEMBLY ID.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.AssemblyServiceApi()
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
assembly_id = 'assembly_id_example' # str | ASSEMBLY ID of the biological assembly candidate.

try:
    # Get structural assembly description by ENTRY ID and ASSEMBLY ID.
    api_response = api_instance.get_assembly_by_id(entry_id, assembly_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling AssemblyServiceApi->get_assembly_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ENTRY ID of the entry. | 
 **assembly_id** | **str**| ASSEMBLY ID of the biological assembly candidate. | 

### Return type

[**CoreAssembly**](CoreAssembly.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

