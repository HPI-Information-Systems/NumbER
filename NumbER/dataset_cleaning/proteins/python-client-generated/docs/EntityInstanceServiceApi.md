# swagger_client.EntityInstanceServiceApi

All URIs are relative to */rest/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_branched_entity_instance_by_id**](EntityInstanceServiceApi.md#get_branched_entity_instance_by_id) | **GET** /core/branched_entity_instance/{entry_id}/{asym_id} | Get branched entity instance description by ENTRY ID and ASYM ID.
[**get_non_polymer_entity_instance_by_id**](EntityInstanceServiceApi.md#get_non_polymer_entity_instance_by_id) | **GET** /core/nonpolymer_entity_instance/{entry_id}/{asym_id} | Get non-polymer entity instance description by ENTRY ID and ASYM ID (label_asym_id).
[**get_polymer_entity_instance_by_id**](EntityInstanceServiceApi.md#get_polymer_entity_instance_by_id) | **GET** /core/polymer_entity_instance/{entry_id}/{asym_id} | Get polymer entity instance (a.k.a chain) data by ENTRY ID and ASYM ID (label_asym_id).

# **get_branched_entity_instance_by_id**
> CorePolymerEntityInstance get_branched_entity_instance_by_id(entry_id, asym_id)

Get branched entity instance description by ENTRY ID and ASYM ID.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.EntityInstanceServiceApi()
entry_id = 'entry_id_example' # str | ID of the entry that needs to be fetched.
asym_id = 'asym_id_example' # str | ID of the instance (chain) that needs to be fetched.

try:
    # Get branched entity instance description by ENTRY ID and ASYM ID.
    api_response = api_instance.get_branched_entity_instance_by_id(entry_id, asym_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EntityInstanceServiceApi->get_branched_entity_instance_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ID of the entry that needs to be fetched. | 
 **asym_id** | **str**| ID of the instance (chain) that needs to be fetched. | 

### Return type

[**CorePolymerEntityInstance**](CorePolymerEntityInstance.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_non_polymer_entity_instance_by_id**
> CoreNonpolymerEntityInstance get_non_polymer_entity_instance_by_id(entry_id, asym_id)

Get non-polymer entity instance description by ENTRY ID and ASYM ID (label_asym_id).

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.EntityInstanceServiceApi()
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
asym_id = 'asym_id_example' # str | ASYM ID (label_asym_id) of the instance (chain).

try:
    # Get non-polymer entity instance description by ENTRY ID and ASYM ID (label_asym_id).
    api_response = api_instance.get_non_polymer_entity_instance_by_id(entry_id, asym_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EntityInstanceServiceApi->get_non_polymer_entity_instance_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ENTRY ID of the entry. | 
 **asym_id** | **str**| ASYM ID (label_asym_id) of the instance (chain). | 

### Return type

[**CoreNonpolymerEntityInstance**](CoreNonpolymerEntityInstance.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_polymer_entity_instance_by_id**
> CorePolymerEntityInstance get_polymer_entity_instance_by_id(entry_id, asym_id)

Get polymer entity instance (a.k.a chain) data by ENTRY ID and ASYM ID (label_asym_id).

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.EntityInstanceServiceApi()
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
asym_id = 'asym_id_example' # str | ASYM ID (label_asym_id) of the instance (chain).

try:
    # Get polymer entity instance (a.k.a chain) data by ENTRY ID and ASYM ID (label_asym_id).
    api_response = api_instance.get_polymer_entity_instance_by_id(entry_id, asym_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EntityInstanceServiceApi->get_polymer_entity_instance_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ENTRY ID of the entry. | 
 **asym_id** | **str**| ASYM ID (label_asym_id) of the instance (chain). | 

### Return type

[**CorePolymerEntityInstance**](CorePolymerEntityInstance.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

