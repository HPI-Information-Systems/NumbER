# EntitySrcGen

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**expression_system_id** | **str** | A unique identifier for the expression system. This  should be extracted from a local list of expression  systems. | [optional] 
**gene_src_common_name** | **str** | The common name of the natural organism from which the gene was  obtained. | [optional] 
**gene_src_details** | **str** | A description of special aspects of the natural organism from  which the gene was obtained. | [optional] 
**gene_src_genus** | **str** | The genus of the natural organism from which the gene was  obtained. | [optional] 
**gene_src_species** | **str** | The species of the natural organism from which the gene was  obtained. | [optional] 
**gene_src_strain** | **str** | The strain of the natural organism from which the gene was  obtained, if relevant. | [optional] 
**gene_src_tissue** | **str** | The tissue of the natural organism from which the gene was  obtained. | [optional] 
**gene_src_tissue_fraction** | **str** | The subcellular fraction of the tissue of the natural organism  from which the gene was obtained. | [optional] 
**host_org_common_name** | **str** | The common name of the organism that served as host for the  production of the entity.  Where full details of the protein  production are available it would be expected that this item  be derived from _entity_src_gen_express.host_org_common_name  or via _entity_src_gen_express.host_org_tax_id | [optional] 
**host_org_details** | **str** | A description of special aspects of the organism that served as  host for the production of the entity. Where full details of  the protein production are available it would be expected that  this item would derived from _entity_src_gen_express.host_org_details | [optional] 
**host_org_genus** | **str** | The genus of the organism that served as host for the production  of the entity. | [optional] 
**host_org_species** | **str** | The species of the organism that served as host for the  production of the entity. | [optional] 
**pdbx_alt_source_flag** | **str** | This data item identifies cases in which an alternative source  modeled. | [optional] 
**pdbx_beg_seq_num** | **int** | The beginning polymer sequence position for the polymer section corresponding  to this source.   A reference to the sequence position in the entity_poly category. | [optional] 
**pdbx_description** | **str** | Information on the source which is not given elsewhere. | [optional] 
**pdbx_end_seq_num** | **int** | The ending polymer sequence position for the polymer section corresponding  to this source.   A reference to the sequence position in the entity_poly category. | [optional] 
**pdbx_gene_src_atcc** | **str** | American Type Culture Collection tissue culture number. | [optional] 
**pdbx_gene_src_cell** | **str** | Cell type. | [optional] 
**pdbx_gene_src_cell_line** | **str** | The specific line of cells. | [optional] 
**pdbx_gene_src_cellular_location** | **str** | Identifies the location inside (or outside) the cell. | [optional] 
**pdbx_gene_src_fragment** | **str** | A domain or fragment of the molecule. | [optional] 
**pdbx_gene_src_gene** | **str** | Identifies the gene. | [optional] 
**pdbx_gene_src_ncbi_taxonomy_id** | **str** | NCBI Taxonomy identifier for the gene source organism.   Reference:   Wheeler DL, Chappey C, Lash AE, Leipe DD, Madden TL, Schuler GD,  Tatusova TA, Rapp BA (2000). Database resources of the National  Center for Biotechnology Information. Nucleic Acids Res 2000 Jan  1;28(1):10-4   Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Rapp BA,  Wheeler DL (2000). GenBank. Nucleic Acids Res 2000 Jan 1;28(1):15-18. | [optional] 
**pdbx_gene_src_organ** | **str** | Organized group of tissues that carries on a specialized function. | [optional] 
**pdbx_gene_src_organelle** | **str** | Organized structure within cell. | [optional] 
**pdbx_gene_src_scientific_name** | **str** | Scientific name of the organism. | [optional] 
**pdbx_gene_src_variant** | **str** | Identifies the variant. | [optional] 
**pdbx_host_org_atcc** | **str** | Americal Tissue Culture Collection of the expression system. Where  full details of the protein production are available it would  be expected that this item  would be derived from  _entity_src_gen_express.host_org_culture_collection | [optional] 
**pdbx_host_org_cell** | **str** | Cell type from which the gene is derived. Where  entity.target_id is provided this should be derived from  details of the target. | [optional] 
**pdbx_host_org_cell_line** | **str** | A specific line of cells used as the expression system. Where  full details of the protein production are available it would  be expected that this item would be derived from  entity_src_gen_express.host_org_cell_line | [optional] 
**pdbx_host_org_cellular_location** | **str** | Identifies the location inside (or outside) the cell which  expressed the molecule. | [optional] 
**pdbx_host_org_culture_collection** | **str** | Culture collection of the expression system. Where  full details of the protein production are available it would  be expected that this item  would be derived somehwere, but  exactly where is not clear. | [optional] 
**pdbx_host_org_gene** | **str** | Specific gene which expressed the molecule. | [optional] 
**pdbx_host_org_ncbi_taxonomy_id** | **str** | NCBI Taxonomy identifier for the expression system organism.   Reference:   Wheeler DL, Chappey C, Lash AE, Leipe DD, Madden TL, Schuler GD,  Tatusova TA, Rapp BA (2000). Database resources of the National  Center for Biotechnology Information. Nucleic Acids Res 2000 Jan  1;28(1):10-4   Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Rapp BA,  Wheeler DL (2000). GenBank. Nucleic Acids Res 2000 Jan 1;28(1):15-18. | [optional] 
**pdbx_host_org_organ** | **str** | Specific organ which expressed the molecule. | [optional] 
**pdbx_host_org_organelle** | **str** | Specific organelle which expressed the molecule. | [optional] 
**pdbx_host_org_scientific_name** | **str** | The scientific name of the organism that served as host for the  production of the entity. Where full details of the protein  production are available it would be expected that this item  would be derived from _entity_src_gen_express.host_org_scientific_name  or via _entity_src_gen_express.host_org_tax_id | [optional] 
**pdbx_host_org_strain** | **str** | The strain of the organism in which the entity was expressed. | [optional] 
**pdbx_host_org_tissue** | **str** | The specific tissue which expressed the molecule. Where full details  of the protein production are available it would be expected that this  item would be derived from _entity_src_gen_express.host_org_tissue | [optional] 
**pdbx_host_org_tissue_fraction** | **str** | The fraction of the tissue which expressed the molecule. | [optional] 
**pdbx_host_org_variant** | **str** | Variant of the organism used as the expression system. Where  full details of the protein production are available it would  be expected that this item be derived from  entity_src_gen_express.host_org_variant or via  _entity_src_gen_express.host_org_tax_id | [optional] 
**pdbx_host_org_vector** | **str** | Identifies the vector used. Where full details of the protein  production are available it would be expected that this item  would be derived from _entity_src_gen_clone.vector_name. | [optional] 
**pdbx_host_org_vector_type** | **str** | Identifies the type of vector used (plasmid, virus, or cosmid).  Where full details of the protein production are available it  would be expected that this item would be derived from  _entity_src_gen_express.vector_type. | [optional] 
**pdbx_seq_type** | **str** | This data item povides additional information about the sequence type. | [optional] 
**pdbx_src_id** | **int** | This data item is an ordinal identifier for entity_src_gen data records. | 
**plasmid_details** | **str** | A description of special aspects of the plasmid that produced the  entity in the host organism. Where full details of the protein  production are available it would be expected that this item  would be derived from _pdbx_construct.details of the construct  pointed to from _entity_src_gen_express.plasmid_id. | [optional] 
**plasmid_name** | **str** | The name of the plasmid that produced the entity in the host  organism. Where full details of the protein production are available  it would be expected that this item would be derived from  _pdbx_construct.name of the construct pointed to from  _entity_src_gen_express.plasmid_id. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
