# swagger_client.EntryServiceApi

All URIs are relative to */rest/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_entry_by_id**](EntryServiceApi.md#get_entry_by_id) | **GET** /core/entry/{entry_id} | Get structure by ENTRY ID.
[**get_pubmed_by_entry_id**](EntryServiceApi.md#get_pubmed_by_entry_id) | **GET** /core/pubmed/{entry_id} | Get PubMed annotations (data integrated from PubMed) for a given entry&#x27;s primary citation.

# **get_entry_by_id**
> CoreEntry get_entry_by_id(entry_id)

Get structure by ENTRY ID.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.EntryServiceApi()
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.

try:
    # Get structure by ENTRY ID.
    api_response = api_instance.get_entry_by_id(entry_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EntryServiceApi->get_entry_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ENTRY ID of the entry. | 

### Return type

[**CoreEntry**](CoreEntry.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_pubmed_by_entry_id**
> CorePubmed get_pubmed_by_entry_id(entry_id)

Get PubMed annotations (data integrated from PubMed) for a given entry's primary citation.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.EntryServiceApi()
entry_id = 'entry_id_example' # str | ENTRY ID of the entry.

try:
    # Get PubMed annotations (data integrated from PubMed) for a given entry's primary citation.
    api_response = api_instance.get_pubmed_by_entry_id(entry_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling EntryServiceApi->get_pubmed_by_entry_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ENTRY ID of the entry. | 

### Return type

[**CorePubmed**](CorePubmed.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

