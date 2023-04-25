# swagger_client.GroupsServiceApi

All URIs are relative to */rest/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_entry_group_by_id**](GroupsServiceApi.md#get_entry_group_by_id) | **GET** /core/entry_groups/{group_id} | PDB cluster data based upon a given aggregation method
[**get_group_provenance_by_id**](GroupsServiceApi.md#get_group_provenance_by_id) | **GET** /core/group_provenance/{group_provenance_id} | Describes aggregation method used to create groups
[**get_polymer_entity_group_by_id**](GroupsServiceApi.md#get_polymer_entity_group_by_id) | **GET** /core/polymer_entity_groups/{group_id} | PDB cluster data based upon a given aggregation method

# **get_entry_group_by_id**
> GroupEntry get_entry_group_by_id(group_id)

PDB cluster data based upon a given aggregation method

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.GroupsServiceApi()
group_id = 'group_id_example' # str | Group ID

try:
    # PDB cluster data based upon a given aggregation method
    api_response = api_instance.get_entry_group_by_id(group_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling GroupsServiceApi->get_entry_group_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **group_id** | **str**| Group ID | 

### Return type

[**GroupEntry**](GroupEntry.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_group_provenance_by_id**
> GroupProvenance get_group_provenance_by_id(group_provenance_id)

Describes aggregation method used to create groups

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.GroupsServiceApi()
group_provenance_id = 'group_provenance_id_example' # str | Group provenance ID

try:
    # Describes aggregation method used to create groups
    api_response = api_instance.get_group_provenance_by_id(group_provenance_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling GroupsServiceApi->get_group_provenance_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **group_provenance_id** | **str**| Group provenance ID | 

### Return type

[**GroupProvenance**](GroupProvenance.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_polymer_entity_group_by_id**
> GroupPolymerEntity get_polymer_entity_group_by_id(group_id)

PDB cluster data based upon a given aggregation method

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.GroupsServiceApi()
group_id = 'group_id_example' # str | Group ID

try:
    # PDB cluster data based upon a given aggregation method
    api_response = api_instance.get_polymer_entity_group_by_id(group_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling GroupsServiceApi->get_polymer_entity_group_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **group_id** | **str**| Group ID | 

### Return type

[**GroupPolymerEntity**](GroupPolymerEntity.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

