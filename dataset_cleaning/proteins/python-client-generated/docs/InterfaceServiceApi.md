# swagger_client.InterfaceServiceApi

All URIs are relative to */rest/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_interface_by_id**](InterfaceServiceApi.md#get_interface_by_id) | **GET** /core/interface/{entry_id}/{assembly_id}/{interface_id} | Get pairwise polymeric interface description by ENTRY ID, ASSEMBLY ID and INTERFACE ID.

# **get_interface_by_id**
> CoreInterface get_interface_by_id(entry_id, assembly_id, interface_id)

Get pairwise polymeric interface description by ENTRY ID, ASSEMBLY ID and INTERFACE ID.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.InterfaceServiceApi()
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
assembly_id = 'assembly_id_example' # str | ASSEMBLY ID of the biological assembly.
interface_id = 'interface_id_example' # str | INTERFACE ID of the pairwise polymeric interface.

try:
    # Get pairwise polymeric interface description by ENTRY ID, ASSEMBLY ID and INTERFACE ID.
    api_response = api_instance.get_interface_by_id(entry_id, assembly_id, interface_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling InterfaceServiceApi->get_interface_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ENTRY ID of the entry. | 
 **assembly_id** | **str**| ASSEMBLY ID of the biological assembly. | 
 **interface_id** | **str**| INTERFACE ID of the pairwise polymeric interface. | 

### Return type

[**CoreInterface**](CoreInterface.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

