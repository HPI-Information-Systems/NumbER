# Cell

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**z_pdb** | **int** | The number of the polymeric chains in a unit cell. In the case  of heteropolymers, Z is the number of occurrences of the most  populous chain.   This data item is provided for compatibility with the original  Protein Data Bank format, and only for that purpose. | [optional] 
**angle_alpha** | **float** | Unit-cell angle alpha of the reported structure in degrees. | [optional] 
**angle_beta** | **float** | Unit-cell angle beta of the reported structure in degrees. | [optional] 
**angle_gamma** | **float** | Unit-cell angle gamma of the reported structure in degrees. | [optional] 
**formula_units_z** | **int** | The number of the formula units in the unit cell as specified  by _chemical_formula.structural, _chemical_formula.moiety or  _chemical_formula.sum. | [optional] 
**length_a** | **float** | Unit-cell length a corresponding to the structure reported in angstroms. | [optional] 
**length_b** | **float** | Unit-cell length b corresponding to the structure reported in  angstroms. | [optional] 
**length_c** | **float** | Unit-cell length c corresponding to the structure reported in angstroms. | [optional] 
**pdbx_unique_axis** | **str** | To further identify unique axis if necessary.  E.g., P 21 with  an unique C axis will have &#x27;C&#x27; in this field. | [optional] 
**volume** | **float** | Cell volume V in angstroms cubed.   V &#x3D; a b c (1 - cos^2^~alpha~ - cos^2^~beta~ - cos^2^~gamma~             + 2 cos~alpha~ cos~beta~ cos~gamma~)^1/2^   a     &#x3D; _cell.length_a  b     &#x3D; _cell.length_b  c     &#x3D; _cell.length_c  alpha &#x3D; _cell.angle_alpha  beta  &#x3D; _cell.angle_beta  gamma &#x3D; _cell.angle_gamma | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

