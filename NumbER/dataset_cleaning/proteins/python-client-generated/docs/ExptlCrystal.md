# ExptlCrystal

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**colour** | **str** | The colour of the crystal. | [optional] 
**density_matthews** | **float** | The density of the crystal, expressed as the ratio of the  volume of the asymmetric unit to the molecular mass of a  monomer of the structure, in units of angstroms^3^ per dalton.   Ref: Matthews, B. W. (1968). J. Mol. Biol. 33, 491-497. | [optional] 
**density_meas** | **float** | Density values measured using standard chemical and physical  methods. The units are megagrams per cubic metre (grams per  cubic centimetre). | [optional] 
**density_percent_sol** | **float** | Density value P calculated from the crystal cell and contents,  expressed as per cent solvent.   P &#x3D; 1 - (1.23 N MMass) / V   N     &#x3D; the number of molecules in the unit cell  MMass &#x3D; the molecular mass of each molecule (gm/mole)  V     &#x3D; the volume of the unit cell (A^3^)  1.23  &#x3D; a conversion factor evaluated as:           (0.74 cm^3^/g) (10^24^ A^3^/cm^3^)          --------------------------------------               (6.02*10^23^) molecules/mole           where 0.74 is an assumed value for the partial specific          volume of the molecule | [optional] 
**description** | **str** | A description of the quality and habit of the crystal.  The crystal dimensions should not normally be reported here;  use instead the specific items in the EXPTL_CRYSTAL category  relating to size for the gross dimensions of the crystal and  data items in the EXPTL_CRYSTAL_FACE category to describe the  relationship between individual faces. | [optional] 
**id** | **str** | The value of _exptl_crystal.id must uniquely identify a record in  the EXPTL_CRYSTAL list.   Note that this item need not be a number; it can be any unique  identifier. | 
**pdbx_mosaicity** | **float** | Isotropic approximation of the distribution of mis-orientation angles specified in degrees of all the mosaic domain blocks in the crystal, represented as a standard deviation. Here, a mosaic block is a set of contiguous unit cells assumed to be perfectly aligned. Lower mosaicity indicates better ordered crystals. See for example:  Nave, C. (1998). Acta Cryst. D54, 848-853.  Note that many software packages estimate the mosaic rotation distribution differently and may combine several physical properties of the experiment into a single mosaic term. This term will help fit the modeled spots to the observed spots without necessarily being directly related to the physics of the crystal itself. | [optional] 
**pdbx_mosaicity_esd** | **float** | The uncertainty in the mosaicity estimate for the crystal. | [optional] 
**preparation** | **str** | Details of crystal growth and preparation of the crystal (e.g.  mounting) prior to the intensity measurements. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

