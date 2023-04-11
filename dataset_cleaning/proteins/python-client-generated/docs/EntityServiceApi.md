# swagger_client.EntityServiceApi

All URIs are relative to */rest/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_branched_entity_by_id**](EntityServiceApi.md#get_branched_entity_by_id) | **GET** /core/branched_entity/{entry_id}/{entity_id} | Get branched entity description by ENTRY ID and ENTITY ID.
[**get_non_polymer_entity_by_id**](EntityServiceApi.md#get_non_polymer_entity_by_id) | **GET** /core/nonpolymer_entity/{entry_id}/{entity_id} | Get non-polymer entity data by ENTRY ID and ENTITY ID.
[**get_polymer_entity_by_id**](EntityServiceApi.md#get_polymer_entity_by_id) | **GET** /core/polymer_entity/{entry_id}/{entity_id} | Get polymer entity data by ENTRY ID and ENTITY ID.
[**get_uniprot_by_entity_id**](EntityServiceApi.md#get_uniprot_by_entity_id) | **GET** /core/uniprot/{entry_id}/{entity_id} | Get UniProt annotations for a given macromolecular entity (identified by ENTRY ID and ENTITY ID).

# **get_branched_entity_by_id**
> CorePolymerEntity get_branched_entity_by_id(entry_id, entity_id)

Get branched entity description by ENTRY ID and ENTITY ID.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.EntityServiceApi()
entry_id = 'entry_id_example' # str | ID of the entry that needs to be fetched.
entity_id = 'entity_id_example' # str | ID of the entity that needs to be fetched.

try:
    # Get branched entity description by ENTRY ID and ENTITY ID.
    api_response = api_instance.get_branched_entity_by_id(entry_id, entity_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EntityServiceApi->get_branched_entity_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ID of the entry that needs to be fetched. | 
 **entity_id** | **str**| ID of the entity that needs to be fetched. | 

### Return type

[**CorePolymerEntity**](CorePolymerEntity.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_non_polymer_entity_by_id**
> CoreNonpolymerEntity get_non_polymer_entity_by_id(entry_id, entity_id)

Get non-polymer entity data by ENTRY ID and ENTITY ID.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.EntityServiceApi()
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
entity_id = 'entity_id_example' # str | ENTITY ID of the non-polymer entity.

try:
    # Get non-polymer entity data by ENTRY ID and ENTITY ID.
    api_response = api_instance.get_non_polymer_entity_by_id(entry_id, entity_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EntityServiceApi->get_non_polymer_entity_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ENTRY ID of the entry. | 
 **entity_id** | **str**| ENTITY ID of the non-polymer entity. | 

### Return type

[**CoreNonpolymerEntity**](CoreNonpolymerEntity.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_polymer_entity_by_id**
> CorePolymerEntity get_polymer_entity_by_id(entry_id, entity_id)

Get polymer entity data by ENTRY ID and ENTITY ID.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.EntityServiceApi()
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
entity_id = 'entity_id_example' # str | ENTITY ID of the polymer entity.

try:
    # Get polymer entity data by ENTRY ID and ENTITY ID.
    api_response = api_instance.get_polymer_entity_by_id(entry_id, entity_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EntityServiceApi->get_polymer_entity_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ENTRY ID of the entry. | 
 **entity_id** | **str**| ENTITY ID of the polymer entity. | 

### Return type

[**CorePolymerEntity**](CorePolymerEntity.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_uniprot_by_entity_id**
> list[CoreUniprot] get_uniprot_by_entity_id(entry_id, entity_id)

Get UniProt annotations for a given macromolecular entity (identified by ENTRY ID and ENTITY ID).

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.EntityServiceApi()
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.
entity_id = 'entity_id_example' # str | ENTITY ID of the polymer entity.

try:
    # Get UniProt annotations for a given macromolecular entity (identified by ENTRY ID and ENTITY ID).
    api_response = api_instance.get_uniprot_by_entity_id(entry_id, entity_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EntityServiceApi->get_uniprot_by_entity_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ENTRY ID of the entry. | 
 **entity_id** | **str**| ENTITY ID of the polymer entity. | 

### Return type

[**list[CoreUniprot]**](CoreUniprot.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

