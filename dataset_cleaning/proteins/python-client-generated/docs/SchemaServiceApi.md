# swagger_client.SchemaServiceApi

All URIs are relative to */rest/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_assembly_metadata**](SchemaServiceApi.md#get_assembly_metadata) | **GET** /schema/assembly | Get assembly schema.
[**get_branched_entity_instance_metadata**](SchemaServiceApi.md#get_branched_entity_instance_metadata) | **GET** /schema/branched_entity_instance | Get branched instance schema.
[**get_branched_entity_metadata**](SchemaServiceApi.md#get_branched_entity_metadata) | **GET** /schema/branched_entity | Get branched entity schema.
[**get_chem_comp_metadata**](SchemaServiceApi.md#get_chem_comp_metadata) | **GET** /schema/chem_comp | Get chemical component/BIRD schema.
[**get_drugbank_metadata**](SchemaServiceApi.md#get_drugbank_metadata) | **GET** /schema/drugbank | Get DrugBank (integrated data) schema.
[**get_entry_metadata**](SchemaServiceApi.md#get_entry_metadata) | **GET** /schema/entry | Get entry schema.
[**get_non_polymer_entity_instance_metadata**](SchemaServiceApi.md#get_non_polymer_entity_instance_metadata) | **GET** /schema/nonpolymer_entity_instance | Get non-polymer instance schema.
[**get_non_polymer_entity_metadata**](SchemaServiceApi.md#get_non_polymer_entity_metadata) | **GET** /schema/nonpolymer_entity | Get non-polymer entity schema.
[**get_polymer_entity_instance_metadata**](SchemaServiceApi.md#get_polymer_entity_instance_metadata) | **GET** /schema/polymer_entity_instance | Get polymer instance schema.
[**get_polymer_entity_metadata**](SchemaServiceApi.md#get_polymer_entity_metadata) | **GET** /schema/polymer_entity | Get polymer entity schema.
[**get_pubmed_metadata**](SchemaServiceApi.md#get_pubmed_metadata) | **GET** /schema/pubmed | Get PubMed (integrated data) schema.
[**get_uniprot_metadata**](SchemaServiceApi.md#get_uniprot_metadata) | **GET** /schema/uniprot | Get UniProt (integrated data) schema.

# **get_assembly_metadata**
> str get_assembly_metadata()

Get assembly schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get assembly schema.
    api_response = api_instance.get_assembly_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_assembly_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_branched_entity_instance_metadata**
> str get_branched_entity_instance_metadata()

Get branched instance schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get branched instance schema.
    api_response = api_instance.get_branched_entity_instance_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_branched_entity_instance_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_branched_entity_metadata**
> str get_branched_entity_metadata()

Get branched entity schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get branched entity schema.
    api_response = api_instance.get_branched_entity_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_branched_entity_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_chem_comp_metadata**
> str get_chem_comp_metadata()

Get chemical component/BIRD schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get chemical component/BIRD schema.
    api_response = api_instance.get_chem_comp_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_chem_comp_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_drugbank_metadata**
> str get_drugbank_metadata()

Get DrugBank (integrated data) schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get DrugBank (integrated data) schema.
    api_response = api_instance.get_drugbank_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_drugbank_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_entry_metadata**
> str get_entry_metadata()

Get entry schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get entry schema.
    api_response = api_instance.get_entry_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_entry_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_non_polymer_entity_instance_metadata**
> str get_non_polymer_entity_instance_metadata()

Get non-polymer instance schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get non-polymer instance schema.
    api_response = api_instance.get_non_polymer_entity_instance_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_non_polymer_entity_instance_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_non_polymer_entity_metadata**
> str get_non_polymer_entity_metadata()

Get non-polymer entity schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get non-polymer entity schema.
    api_response = api_instance.get_non_polymer_entity_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_non_polymer_entity_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_polymer_entity_instance_metadata**
> str get_polymer_entity_instance_metadata()

Get polymer instance schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get polymer instance schema.
    api_response = api_instance.get_polymer_entity_instance_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_polymer_entity_instance_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_polymer_entity_metadata**
> str get_polymer_entity_metadata()

Get polymer entity schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get polymer entity schema.
    api_response = api_instance.get_polymer_entity_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_polymer_entity_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_pubmed_metadata**
> str get_pubmed_metadata()

Get PubMed (integrated data) schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get PubMed (integrated data) schema.
    api_response = api_instance.get_pubmed_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_pubmed_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_uniprot_metadata**
> str get_uniprot_metadata()

Get UniProt (integrated data) schema.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.SchemaServiceApi()

try:
    # Get UniProt (integrated data) schema.
    api_response = api_instance.get_uniprot_metadata()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SchemaServiceApi->get_uniprot_metadata: %s\n" % e)
```

### Parameters
This endpoint does not need any parameter.

### Return type

**str**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

