# PdbxStructAssembly

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**details** | **str** | A description of special aspects of the macromolecular assembly.                 In the PDB, &#x27;representative helical assembly&#x27;, &#x27;complete point assembly&#x27;,         &#x27;complete icosahedral assembly&#x27;, &#x27;software_defined_assembly&#x27;, &#x27;author_defined_assembly&#x27;,         and &#x27;author_and_software_defined_assembly&#x27; are considered \&quot;biologically relevant assemblies. | [optional] 
**id** | **str** | The value of _pdbx_struct_assembly.id must uniquely identify a record in  the PDBX_STRUCT_ASSEMBLY list. | 
**method_details** | **str** | Provides details of the method used to determine or  compute the assembly. | [optional] 
**oligomeric_count** | **int** | The number of polymer molecules in the assembly. | [optional] 
**oligomeric_details** | **str** | Provides the details of the oligomeric state of the assembly. | [optional] 
**rcsb_candidate_assembly** | **str** | Candidate macromolecular assembly.   Excludes the following cases classified in pdbx_struct_asembly.details:   &#x27;crystal asymmetric unit&#x27;, &#x27;crystal asymmetric unit, crystal frame&#x27;, &#x27;helical asymmetric unit&#x27;,  &#x27;helical asymmetric unit, std helical frame&#x27;,&#x27;icosahedral 23 hexamer&#x27;, &#x27;icosahedral asymmetric unit&#x27;,  &#x27;icosahedral asymmetric unit, std point frame&#x27;,&#x27;icosahedral pentamer&#x27;, &#x27;pentasymmetron capsid unit&#x27;,  &#x27;point asymmetric unit&#x27;, &#x27;point asymmetric unit, std point frame&#x27;,&#x27;trisymmetron capsid unit&#x27;,   and &#x27;deposited_coordinates&#x27;. | [optional] 
**rcsb_details** | **str** | A filtered description of the macromolecular assembly. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

