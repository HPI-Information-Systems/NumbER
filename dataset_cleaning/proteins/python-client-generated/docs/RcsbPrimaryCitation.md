# RcsbPrimaryCitation

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**book_id_isbn** | **str** | The International Standard Book Number (ISBN) code assigned to  the book cited; relevant for books or book chapters. | [optional] 
**book_publisher** | **str** | The name of the publisher of the citation; relevant  for books or book chapters. | [optional] 
**book_publisher_city** | **str** | The location of the publisher of the citation; relevant  for books or book chapters. | [optional] 
**book_title** | **str** | The title of the book in which the citation appeared; relevant  for books or book chapters. | [optional] 
**coordinate_linkage** | **str** | _rcsb_primary_citation.coordinate_linkage states whether this citation  is concerned with precisely the set of coordinates given in the  data block. If, for instance, the publication described the same  structure, but the coordinates had undergone further refinement  prior to the creation of the data block, the value of this data  item would be &#x27;no&#x27;. | [optional] 
**country** | **str** | The country/region of publication; relevant for books  and book chapters. | [optional] 
**id** | **str** | The value of _rcsb_primary_citation.id must uniquely identify a record in the  CITATION list.   The _rcsb_primary_citation.id &#x27;primary&#x27; should be used to indicate the  citation that the author(s) consider to be the most pertinent to  the contents of the data block.   Note that this item need not be a number; it can be any unique  identifier. | 
**journal_abbrev** | **str** | Abbreviated name of the cited journal as given in the  Chemical Abstracts Service Source Index. | [optional] 
**journal_id_astm** | **str** | The American Society for Testing and Materials (ASTM) code  assigned to the journal cited (also referred to as the CODEN  designator of the Chemical Abstracts Service); relevant for  journal articles. | [optional] 
**journal_id_csd** | **str** | The Cambridge Structural Database (CSD) code assigned to the  journal cited; relevant for journal articles. This is also the  system used at the Protein Data Bank (PDB). | [optional] 
**journal_id_issn** | **str** | The International Standard Serial Number (ISSN) code assigned to  the journal cited; relevant for journal articles. | [optional] 
**journal_issue** | **str** | Issue number of the journal cited; relevant for journal  articles. | [optional] 
**journal_volume** | **str** | Volume number of the journal cited; relevant for journal  articles. | [optional] 
**language** | **str** | Language in which the cited article is written. | [optional] 
**page_first** | **str** | The first page of the citation; relevant for journal  articles, books and book chapters. | [optional] 
**page_last** | **str** | The last page of the citation; relevant for journal  articles, books and book chapters. | [optional] 
**pdbx_database_id_doi** | **str** | Document Object Identifier used by doi.org to uniquely  specify bibliographic entry. | [optional] 
**pdbx_database_id_pub_med** | **int** | Ascession number used by PubMed to categorize a specific  bibliographic entry. | [optional] 
**rcsb_orcid_identifiers** | **list[str]** | The Open Researcher and Contributor ID (ORCID) identifiers for the citation authors. | [optional] 
**rcsb_authors** | **list[str]** | Names of the authors of the citation; relevant for journal  articles, books and book chapters.  Names are separated by vertical bars.   The family name(s), followed by a comma and including any  dynastic components, precedes the first name(s) or initial(s). | [optional] 
**rcsb_journal_abbrev** | **str** | Normalized journal abbreviation. | [optional] 
**title** | **str** | The title of the citation; relevant for journal articles, books  and book chapters. | [optional] 
**year** | **int** | The year of the citation; relevant for journal articles, books  and book chapters. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)

