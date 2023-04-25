# EntityPoly

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**nstd_linkage** | **str** | A flag to indicate whether the polymer contains at least  one monomer-to-monomer link different from that implied by  _entity_poly.type. | [optional] 
**nstd_monomer** | **str** | A flag to indicate whether the polymer contains at least  one monomer that is not considered standard. | [optional] 
**pdbx_seq_one_letter_code** | **str** | Sequence of protein or nucleic acid polymer in standard one-letter                codes of amino acids or nucleotides. Non-standard amino                acids/nucleotides are represented by their Chemical                Component Dictionary (CCD) codes in                parenthesis. Deoxynucleotides are represented by the                specially-assigned 2-letter CCD codes in parenthesis,                with &#x27;D&#x27; prefix added to their ribonucleotide                counterparts. For hybrid polymer, each residue is                represented by the code of its individual type. A                cyclic polymer is represented in linear sequence from                the chosen start to end.  A for Alanine or Adenosine-5&#x27;-monophosphate C for Cysteine or Cytidine-5&#x27;-monophosphate D for Aspartic acid E for Glutamic acid F for Phenylalanine G for Glycine or Guanosine-5&#x27;-monophosphate H for Histidine I for Isoleucine or Inosinic Acid L for Leucine K for Lysine M for Methionine N for Asparagine  or Unknown ribonucleotide O for Pyrrolysine P for Proline Q for Glutamine R for Arginine S for Serine T for Threonine U for Selenocysteine or Uridine-5&#x27;-monophosphate V for Valine W for Tryptophan Y for Tyrosine (DA) for 2&#x27;-deoxyadenosine-5&#x27;-monophosphate (DC) for 2&#x27;-deoxycytidine-5&#x27;-monophosphate (DG) for 2&#x27;-deoxyguanosine-5&#x27;-monophosphate (DT) for Thymidine-5&#x27;-monophosphate (MSE) for Selenomethionine (SEP) for Phosphoserine (PTO) for Phosphothreonine (PTR) for Phosphotyrosine (PCA) for Pyroglutamic acid (UNK) for Unknown amino acid (ACE) for Acetylation cap (NH2) for Amidation cap | [optional] 
**pdbx_seq_one_letter_code_can** | **str** | Canonical sequence of protein or nucleic acid polymer in standard                one-letter codes of amino acids or nucleotides,                corresponding to the sequence in                _entity_poly.pdbx_seq_one_letter_code. Non-standard                amino acids/nucleotides are represented by the codes of                their parents if parent is specified in                _chem_comp.mon_nstd_parent_comp_id, or by letter &#x27;X&#x27; if                parent is not specified. Deoxynucleotides are                represented by their canonical one-letter codes of A,                C, G, or T.                 For modifications with several parent amino acids,         all corresponding parent amino acid codes will be listed         (ex. chromophores). | [optional] 
**pdbx_sequence_evidence_code** | **str** | Evidence for the assignment of the polymer sequence. | [optional] 
**pdbx_strand_id** | **str** | The PDB strand/chain id(s) corresponding to this polymer entity. | [optional] 
**pdbx_target_identifier** | **str** | For Structural Genomics entries, the sequence&#x27;s target identifier registered at the TargetTrack database. | [optional] 
**rcsb_artifact_monomer_count** | **int** | Number of regions in the sample sequence identified as expression tags, linkers, or  cloning artifacts. | [optional] 
**rcsb_conflict_count** | **int** | Number of monomer conflicts relative to the reference sequence. | [optional] 
**rcsb_deletion_count** | **int** | Number of monomer deletions relative to the reference sequence. | [optional] 
**rcsb_entity_polymer_type** | **str** | A coarse-grained polymer entity type. | [optional] 
**rcsb_insertion_count** | **int** | Number of monomer insertions relative to the reference sequence. | [optional] 
**rcsb_mutation_count** | **int** | Number of engineered mutations engineered in the sample sequence. | [optional] 
**rcsb_non_std_monomer_count** | **int** | Number of non-standard monomers in the sample sequence. | [optional] 
**rcsb_non_std_monomers** | **list[str]** | Unique list of non-standard monomer chemical component identifiers in the sample sequence. | [optional] 
**rcsb_prd_id** | **str** | For polymer BIRD molecules the BIRD identifier for the entity. | [optional] 
**rcsb_sample_sequence_length** | **int** | The monomer length of the sample sequence. | [optional] 
**type** | **str** | The type of the polymer. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

