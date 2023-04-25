# RcsbStructSymmetry

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**symbol** | **str** | Symmetry symbol refers to point group or helical symmetry of identical polymeric subunits in Sch�nflies notation. Contains point group symbol (e.g., C2, C5, D2, T, O, I) or H for helical symmetry. | 
**type** | **str** | Symmetry type refers to point group or helical symmetry of identical polymeric subunits. Contains point group types (e.g. Cyclic, Dihedral) or Helical for helical symmetry. | 
**stoichiometry** | **list[str]** | Each type of different subunits is assigned a latter. The number of equivalent subunits is added as a coefficient after each letter (except 1 which is not added explicitly). | 
**oligomeric_state** | **str** | Oligomeric state refers to a composition of polymeric subunits in quaternary structure. Quaternary structure may be composed either exclusively of several copies of identical subunits, in which case they are termed homo-oligomers, or alternatively by at least one copy of different subunits (hetero-oligomers). Quaternary structure composed of a single subunit is denoted as &#x27;Monomer&#x27;. | 
**clusters** | [**list[RcsbStructSymmetryClusters]**](RcsbStructSymmetryClusters.md) |  | 
**rotation_axes** | [**list[RcsbStructSymmetryRotationAxes]**](RcsbStructSymmetryRotationAxes.md) | The orientation of the principal rotation (symmetry) axis. | [optional] 
**kind** | **str** | The granularity at which the symmetry calculation is performed. In &#x27;Global Symmetry&#x27; all polymeric subunits in assembly are used. In &#x27;Local Symmetry&#x27; only a subset of polymeric subunits is considered. In &#x27;Pseudo Symmetry&#x27; the threshold for subunits similarity is relaxed. | 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

