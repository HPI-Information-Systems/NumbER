# swagger_client.RepositoryHoldingsServiceApi

All URIs are relative to */rest/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_current_entry_ids**](RepositoryHoldingsServiceApi.md#get_current_entry_ids) | **GET** /holdings/current/entry_ids | Get the list of current entry IDs.
[**get_entry_status**](RepositoryHoldingsServiceApi.md#get_entry_status) | **GET** /holdings/status/{entry_id} | Get the status and status code of a structure.
[**get_entry_statuses**](RepositoryHoldingsServiceApi.md#get_entry_statuses) | **GET** /holdings/status | Get the status and status code of a list of entry IDs.
[**get_removed_entry_by_id**](RepositoryHoldingsServiceApi.md#get_removed_entry_by_id) | **GET** /holdings/removed/{entry_id} | Get the description of the structure that was removed from the PDB repository.
[**get_removed_entry_ids**](RepositoryHoldingsServiceApi.md#get_removed_entry_ids) | **GET** /holdings/removed/entry_ids | Get the list of removed entry IDs.
[**get_unreleased_entries**](RepositoryHoldingsServiceApi.md#get_unreleased_entries) | **GET** /holdings/unreleased | Get a list of unreleased structures with descriptions.
[**get_unreleased_entry_by_id**](RepositoryHoldingsServiceApi.md#get_unreleased_entry_by_id) | **GET** /holdings/unreleased/{entry_id} | Get the description of unreleased structure that is being processed or on hold waiting for release.
[**get_unreleased_entry_ids**](RepositoryHoldingsServiceApi.md#get_unreleased_entry_ids) | **GET** /holdings/unreleased/entry_ids | Get the list of unreleased entry IDs.

# **get_current_entry_ids**
> list[str] get_current_entry_ids()

Get the list of current entry IDs.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.RepositoryHoldingsServiceApi()

try:
    # Get the list of current entry IDs.
    api_response = api_instance.get_current_entry_ids()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling RepositoryHoldingsServiceApi->get_current_entry_ids: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**list[str]**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_entry_status**
> CombinedEntry get_entry_status(entry_id)

Get the status and status code of a structure.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.RepositoryHoldingsServiceApi()
entry_id = 'entry_id_example' # str | ID of the entry that needs to be fetched.

try:
    # Get the status and status code of a structure.
    api_response = api_instance.get_entry_status(entry_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling RepositoryHoldingsServiceApi->get_entry_status: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ID of the entry that needs to be fetched. | 

### Return type

[**CombinedEntry**](CombinedEntry.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_entry_statuses**
> list[CombinedEntry] get_entry_statuses(ids)

Get the status and status code of a list of entry IDs.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.RepositoryHoldingsServiceApi()
ids = 'ids_example' # str | A comma separated entry ID list.

try:
    # Get the status and status code of a list of entry IDs.
    api_response = api_instance.get_entry_statuses(ids)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling RepositoryHoldingsServiceApi->get_entry_statuses: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **ids** | **str**| A comma separated entry ID list. | 

### Return type

[**list[CombinedEntry]**](CombinedEntry.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_removed_entry_by_id**
> RemovedEntry get_removed_entry_by_id(entry_id)

Get the description of the structure that was removed from the PDB repository.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.RepositoryHoldingsServiceApi()
entry_id = 'entry_id_example' # str | ID of the entry that needs to be fetched.

try:
    # Get the description of the structure that was removed from the PDB repository.
    api_response = api_instance.get_removed_entry_by_id(entry_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling RepositoryHoldingsServiceApi->get_removed_entry_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ID of the entry that needs to be fetched. | 

### Return type

[**RemovedEntry**](RemovedEntry.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_removed_entry_ids**
> list[str] get_removed_entry_ids()

Get the list of removed entry IDs.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.RepositoryHoldingsServiceApi()

try:
    # Get the list of removed entry IDs.
    api_response = api_instance.get_removed_entry_ids()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling RepositoryHoldingsServiceApi->get_removed_entry_ids: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**list[str]**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_unreleased_entries**
> list[UnreleasedEntry] get_unreleased_entries(ids)

Get a list of unreleased structures with descriptions.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.RepositoryHoldingsServiceApi()
ids = 'ids_example' # str | A comma separated unreleased entry ID list.

try:
    # Get a list of unreleased structures with descriptions.
    api_response = api_instance.get_unreleased_entries(ids)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling RepositoryHoldingsServiceApi->get_unreleased_entries: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **ids** | **str**| A comma separated unreleased entry ID list. | 

### Return type

[**list[UnreleasedEntry]**](UnreleasedEntry.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_unreleased_entry_by_id**
> UnreleasedEntry get_unreleased_entry_by_id(entry_id)

Get the description of unreleased structure that is being processed or on hold waiting for release.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.RepositoryHoldingsServiceApi()
entry_id = 'entry_id_example' # str | ID of the entry that needs to be fetched.

try:
    # Get the description of unreleased structure that is being processed or on hold waiting for release.
    api_response = api_instance.get_unreleased_entry_by_id(entry_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling RepositoryHoldingsServiceApi->get_unreleased_entry_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **entry_id** | **str**| ID of the entry that needs to be fetched. | 

### Return type

[**UnreleasedEntry**](UnreleasedEntry.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_unreleased_entry_ids**
> list[str] get_unreleased_entry_ids()

Get the list of unreleased entry IDs.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.RepositoryHoldingsServiceApi()

try:
    # Get the list of unreleased entry IDs.
    api_response = api_instance.get_unreleased_entry_ids()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling RepositoryHoldingsServiceApi->get_unreleased_entry_ids: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**list[str]**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

