# swagger_client.ChemicalComponentServiceApi

All URIs are relative to */rest/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_chem_comp_by_id**](ChemicalComponentServiceApi.md#get_chem_comp_by_id) | **GET** /core/chemcomp/{comp_id} | Get chemical component (ligands, small molecules and monomers) by CCD ID.
[**get_drug_bank_by_chem_comp_id**](ChemicalComponentServiceApi.md#get_drug_bank_by_chem_comp_id) | **GET** /core/drugbank/{comp_id} | Get DrugBank annotations (integrated from DrugBank resource) for a given chemical component (identified by CHEM COMP ID) .

# **get_chem_comp_by_id**
> CoreChemComp get_chem_comp_by_id(comp_id)

Get chemical component (ligands, small molecules and monomers) by CCD ID.

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.ChemicalComponentServiceApi()
comp_id = 'comp_id_example' # str | CHEM COMP ID that uniquely identifies the chemical component.  For protein polymer entities, this is the three-letter code for the amino acid.  For nucleic acid polymer entities, this is the one-letter code for the base.

try:
    # Get chemical component (ligands, small molecules and monomers) by CCD ID.
    api_response = api_instance.get_chem_comp_by_id(comp_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ChemicalComponentServiceApi->get_chem_comp_by_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **comp_id** | **str**| CHEM COMP ID that uniquely identifies the chemical component.  For protein polymer entities, this is the three-letter code for the amino acid.  For nucleic acid polymer entities, this is the one-letter code for the base. | 

### Return type

[**CoreChemComp**](CoreChemComp.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_drug_bank_by_chem_comp_id**
> CoreDrugbank get_drug_bank_by_chem_comp_id(comp_id)

Get DrugBank annotations (integrated from DrugBank resource) for a given chemical component (identified by CHEM COMP ID) .

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.ChemicalComponentServiceApi()
comp_id = 'comp_id_example' # str | CHEM COMP ID that uniquely identifies the chemical component.  For protein polymer entities, this is the three-letter code for the amino acid.  For nucleic acid polymer entities, this is the one-letter code for the base.

try:
    # Get DrugBank annotations (integrated from DrugBank resource) for a given chemical component (identified by CHEM COMP ID) .
    api_response = api_instance.get_drug_bank_by_chem_comp_id(comp_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling ChemicalComponentServiceApi->get_drug_bank_by_chem_comp_id: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **comp_id** | **str**| CHEM COMP ID that uniquely identifies the chemical component.  For protein polymer entities, this is the three-letter code for the amino acid.  For nucleic acid polymer entities, this is the one-letter code for the base. | 

### Return type

[**CoreDrugbank**](CoreDrugbank.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json;charset=utf-8

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

