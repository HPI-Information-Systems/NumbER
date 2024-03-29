# coding: utf-8

"""
    RCSB RESTful API

    Provides programmatic access to information and annotations stored in the Protein Data Bank. <br>Models are generated from JSON schema version: <b>1.40.0</b>. <br>API services deployed on: Sun, 2 Apr 2023 21:44:46 -0700  # noqa: E501

    OpenAPI spec version: 1.40.0
    Contact: info@rcsb.org
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six

class RcsbEntityHostOrganism(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'beg_seq_num': 'int',
        'common_name': 'str',
        'end_seq_num': 'int',
        'ncbi_common_names': 'list[str]',
        'ncbi_parent_scientific_name': 'str',
        'ncbi_scientific_name': 'str',
        'ncbi_taxonomy_id': 'int',
        'pdbx_src_id': 'str',
        'provenance_source': 'str',
        'scientific_name': 'str',
        'taxonomy_lineage': 'list[RcsbEntityHostOrganismTaxonomyLineage]'
    }

    attribute_map = {
        'beg_seq_num': 'beg_seq_num',
        'common_name': 'common_name',
        'end_seq_num': 'end_seq_num',
        'ncbi_common_names': 'ncbi_common_names',
        'ncbi_parent_scientific_name': 'ncbi_parent_scientific_name',
        'ncbi_scientific_name': 'ncbi_scientific_name',
        'ncbi_taxonomy_id': 'ncbi_taxonomy_id',
        'pdbx_src_id': 'pdbx_src_id',
        'provenance_source': 'provenance_source',
        'scientific_name': 'scientific_name',
        'taxonomy_lineage': 'taxonomy_lineage'
    }

    def __init__(self, beg_seq_num=None, common_name=None, end_seq_num=None, ncbi_common_names=None, ncbi_parent_scientific_name=None, ncbi_scientific_name=None, ncbi_taxonomy_id=None, pdbx_src_id=None, provenance_source=None, scientific_name=None, taxonomy_lineage=None):  # noqa: E501
        """RcsbEntityHostOrganism - a model defined in Swagger"""  # noqa: E501
        self._beg_seq_num = None
        self._common_name = None
        self._end_seq_num = None
        self._ncbi_common_names = None
        self._ncbi_parent_scientific_name = None
        self._ncbi_scientific_name = None
        self._ncbi_taxonomy_id = None
        self._pdbx_src_id = None
        self._provenance_source = None
        self._scientific_name = None
        self._taxonomy_lineage = None
        self.discriminator = None
        if beg_seq_num is not None:
            self.beg_seq_num = beg_seq_num
        if common_name is not None:
            self.common_name = common_name
        if end_seq_num is not None:
            self.end_seq_num = end_seq_num
        if ncbi_common_names is not None:
            self.ncbi_common_names = ncbi_common_names
        if ncbi_parent_scientific_name is not None:
            self.ncbi_parent_scientific_name = ncbi_parent_scientific_name
        if ncbi_scientific_name is not None:
            self.ncbi_scientific_name = ncbi_scientific_name
        if ncbi_taxonomy_id is not None:
            self.ncbi_taxonomy_id = ncbi_taxonomy_id
        self.pdbx_src_id = pdbx_src_id
        if provenance_source is not None:
            self.provenance_source = provenance_source
        if scientific_name is not None:
            self.scientific_name = scientific_name
        if taxonomy_lineage is not None:
            self.taxonomy_lineage = taxonomy_lineage

    @property
    def beg_seq_num(self):
        """Gets the beg_seq_num of this RcsbEntityHostOrganism.  # noqa: E501

        The beginning polymer sequence position for the polymer section corresponding  to this host organism.   A reference to the sequence position in the entity_poly category.  # noqa: E501

        :return: The beg_seq_num of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: int
        """
        return self._beg_seq_num

    @beg_seq_num.setter
    def beg_seq_num(self, beg_seq_num):
        """Sets the beg_seq_num of this RcsbEntityHostOrganism.

        The beginning polymer sequence position for the polymer section corresponding  to this host organism.   A reference to the sequence position in the entity_poly category.  # noqa: E501

        :param beg_seq_num: The beg_seq_num of this RcsbEntityHostOrganism.  # noqa: E501
        :type: int
        """

        self._beg_seq_num = beg_seq_num

    @property
    def common_name(self):
        """Gets the common_name of this RcsbEntityHostOrganism.  # noqa: E501

        The common name of the host organism  # noqa: E501

        :return: The common_name of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: str
        """
        return self._common_name

    @common_name.setter
    def common_name(self, common_name):
        """Sets the common_name of this RcsbEntityHostOrganism.

        The common name of the host organism  # noqa: E501

        :param common_name: The common_name of this RcsbEntityHostOrganism.  # noqa: E501
        :type: str
        """

        self._common_name = common_name

    @property
    def end_seq_num(self):
        """Gets the end_seq_num of this RcsbEntityHostOrganism.  # noqa: E501

        The ending polymer sequence position for the polymer section corresponding  to this host organism.   A reference to the sequence position in the entity_poly category.  # noqa: E501

        :return: The end_seq_num of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: int
        """
        return self._end_seq_num

    @end_seq_num.setter
    def end_seq_num(self, end_seq_num):
        """Sets the end_seq_num of this RcsbEntityHostOrganism.

        The ending polymer sequence position for the polymer section corresponding  to this host organism.   A reference to the sequence position in the entity_poly category.  # noqa: E501

        :param end_seq_num: The end_seq_num of this RcsbEntityHostOrganism.  # noqa: E501
        :type: int
        """

        self._end_seq_num = end_seq_num

    @property
    def ncbi_common_names(self):
        """Gets the ncbi_common_names of this RcsbEntityHostOrganism.  # noqa: E501

        Common names associated with this taxonomy code obtained from NCBI Taxonomy Database.    These names correspond to the taxonomy identifier assigned by the PDB depositor.  References:  Sayers EW, Barrett T, Benson DA, Bryant SH, Canese K, Chetvernin V, Church DM, DiCuccio M, Edgar R, Federhen S, Feolo M, Geer LY, Helmberg W, Kapustin Y, Landsman D, Lipman DJ, Madden TL, Maglott DR, Miller V, Mizrachi I, Ostell J, Pruitt KD, Schuler GD, Sequeira E, Sherry ST, Shumway M, Sirotkin K, Souvorov A, Starchenko G, Tatusova TA, Wagner L, Yaschenko E, Ye J (2009). Database resources of the National Center for Biotechnology Information. Nucleic Acids Res. 2009 Jan;37(Database issue):D5-15. Epub 2008 Oct 21.  Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Sayers EW (2009). GenBank. Nucleic Acids Res. 2009 Jan;37(Database issue):D26-31. Epub 2008 Oct 21.  # noqa: E501

        :return: The ncbi_common_names of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: list[str]
        """
        return self._ncbi_common_names

    @ncbi_common_names.setter
    def ncbi_common_names(self, ncbi_common_names):
        """Sets the ncbi_common_names of this RcsbEntityHostOrganism.

        Common names associated with this taxonomy code obtained from NCBI Taxonomy Database.    These names correspond to the taxonomy identifier assigned by the PDB depositor.  References:  Sayers EW, Barrett T, Benson DA, Bryant SH, Canese K, Chetvernin V, Church DM, DiCuccio M, Edgar R, Federhen S, Feolo M, Geer LY, Helmberg W, Kapustin Y, Landsman D, Lipman DJ, Madden TL, Maglott DR, Miller V, Mizrachi I, Ostell J, Pruitt KD, Schuler GD, Sequeira E, Sherry ST, Shumway M, Sirotkin K, Souvorov A, Starchenko G, Tatusova TA, Wagner L, Yaschenko E, Ye J (2009). Database resources of the National Center for Biotechnology Information. Nucleic Acids Res. 2009 Jan;37(Database issue):D5-15. Epub 2008 Oct 21.  Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Sayers EW (2009). GenBank. Nucleic Acids Res. 2009 Jan;37(Database issue):D26-31. Epub 2008 Oct 21.  # noqa: E501

        :param ncbi_common_names: The ncbi_common_names of this RcsbEntityHostOrganism.  # noqa: E501
        :type: list[str]
        """

        self._ncbi_common_names = ncbi_common_names

    @property
    def ncbi_parent_scientific_name(self):
        """Gets the ncbi_parent_scientific_name of this RcsbEntityHostOrganism.  # noqa: E501

        The parent scientific name in the NCBI taxonomy hierarchy (depth=1) associated with this taxonomy code.  References:  Sayers EW, Barrett T, Benson DA, Bryant SH, Canese K, Chetvernin V, Church DM, DiCuccio M, Edgar R, Federhen S, Feolo M, Geer LY, Helmberg W, Kapustin Y, Landsman D, Lipman DJ, Madden TL, Maglott DR, Miller V, Mizrachi I, Ostell J, Pruitt KD, Schuler GD, Sequeira E, Sherry ST, Shumway M, Sirotkin K, Souvorov A, Starchenko G, Tatusova TA, Wagner L, Yaschenko E, Ye J (2009). Database resources of the National Center for Biotechnology Information. Nucleic Acids Res. 2009 Jan;37(Database issue):D5-15. Epub 2008 Oct 21.  Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Sayers EW (2009). GenBank. Nucleic Acids Res. 2009 Jan;37(Database issue):D26-31. Epub 2008 Oct 21.  # noqa: E501

        :return: The ncbi_parent_scientific_name of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: str
        """
        return self._ncbi_parent_scientific_name

    @ncbi_parent_scientific_name.setter
    def ncbi_parent_scientific_name(self, ncbi_parent_scientific_name):
        """Sets the ncbi_parent_scientific_name of this RcsbEntityHostOrganism.

        The parent scientific name in the NCBI taxonomy hierarchy (depth=1) associated with this taxonomy code.  References:  Sayers EW, Barrett T, Benson DA, Bryant SH, Canese K, Chetvernin V, Church DM, DiCuccio M, Edgar R, Federhen S, Feolo M, Geer LY, Helmberg W, Kapustin Y, Landsman D, Lipman DJ, Madden TL, Maglott DR, Miller V, Mizrachi I, Ostell J, Pruitt KD, Schuler GD, Sequeira E, Sherry ST, Shumway M, Sirotkin K, Souvorov A, Starchenko G, Tatusova TA, Wagner L, Yaschenko E, Ye J (2009). Database resources of the National Center for Biotechnology Information. Nucleic Acids Res. 2009 Jan;37(Database issue):D5-15. Epub 2008 Oct 21.  Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Sayers EW (2009). GenBank. Nucleic Acids Res. 2009 Jan;37(Database issue):D26-31. Epub 2008 Oct 21.  # noqa: E501

        :param ncbi_parent_scientific_name: The ncbi_parent_scientific_name of this RcsbEntityHostOrganism.  # noqa: E501
        :type: str
        """

        self._ncbi_parent_scientific_name = ncbi_parent_scientific_name

    @property
    def ncbi_scientific_name(self):
        """Gets the ncbi_scientific_name of this RcsbEntityHostOrganism.  # noqa: E501

        The scientific name associated with this taxonomy code aggregated by the NCBI Taxonomy Database.    This name corresponds to the taxonomy identifier assigned by the PDB depositor.   References:  Sayers EW, Barrett T, Benson DA, Bryant SH, Canese K, Chetvernin V, Church DM, DiCuccio M, Edgar R, Federhen S, Feolo M, Geer LY, Helmberg W, Kapustin Y, Landsman D, Lipman DJ, Madden TL, Maglott DR, Miller V, Mizrachi I, Ostell J, Pruitt KD, Schuler GD, Sequeira E, Sherry ST, Shumway M, Sirotkin K, Souvorov A, Starchenko G, Tatusova TA, Wagner L, Yaschenko E, Ye J (2009). Database resources of the National Center for Biotechnology Information. Nucleic Acids Res. 2009 Jan;37(Database issue):D5-15. Epub 2008 Oct 21.  Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Sayers EW (2009). GenBank. Nucleic Acids Res. 2009 Jan;37(Database issue):D26-31. Epub 2008 Oct 21.  # noqa: E501

        :return: The ncbi_scientific_name of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: str
        """
        return self._ncbi_scientific_name

    @ncbi_scientific_name.setter
    def ncbi_scientific_name(self, ncbi_scientific_name):
        """Sets the ncbi_scientific_name of this RcsbEntityHostOrganism.

        The scientific name associated with this taxonomy code aggregated by the NCBI Taxonomy Database.    This name corresponds to the taxonomy identifier assigned by the PDB depositor.   References:  Sayers EW, Barrett T, Benson DA, Bryant SH, Canese K, Chetvernin V, Church DM, DiCuccio M, Edgar R, Federhen S, Feolo M, Geer LY, Helmberg W, Kapustin Y, Landsman D, Lipman DJ, Madden TL, Maglott DR, Miller V, Mizrachi I, Ostell J, Pruitt KD, Schuler GD, Sequeira E, Sherry ST, Shumway M, Sirotkin K, Souvorov A, Starchenko G, Tatusova TA, Wagner L, Yaschenko E, Ye J (2009). Database resources of the National Center for Biotechnology Information. Nucleic Acids Res. 2009 Jan;37(Database issue):D5-15. Epub 2008 Oct 21.  Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Sayers EW (2009). GenBank. Nucleic Acids Res. 2009 Jan;37(Database issue):D26-31. Epub 2008 Oct 21.  # noqa: E501

        :param ncbi_scientific_name: The ncbi_scientific_name of this RcsbEntityHostOrganism.  # noqa: E501
        :type: str
        """

        self._ncbi_scientific_name = ncbi_scientific_name

    @property
    def ncbi_taxonomy_id(self):
        """Gets the ncbi_taxonomy_id of this RcsbEntityHostOrganism.  # noqa: E501

        NCBI Taxonomy identifier for the host organism.    Reference:   Wheeler DL, Chappey C, Lash AE, Leipe DD, Madden TL, Schuler GD,  Tatusova TA, Rapp BA (2000). Database resources of the National  Center for Biotechnology Information. Nucleic Acids Res 2000 Jan  1;28(1):10-4   Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Rapp BA,  Wheeler DL (2000). GenBank. Nucleic Acids Res 2000 Jan 1;28(1):15-18.  # noqa: E501

        :return: The ncbi_taxonomy_id of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: int
        """
        return self._ncbi_taxonomy_id

    @ncbi_taxonomy_id.setter
    def ncbi_taxonomy_id(self, ncbi_taxonomy_id):
        """Sets the ncbi_taxonomy_id of this RcsbEntityHostOrganism.

        NCBI Taxonomy identifier for the host organism.    Reference:   Wheeler DL, Chappey C, Lash AE, Leipe DD, Madden TL, Schuler GD,  Tatusova TA, Rapp BA (2000). Database resources of the National  Center for Biotechnology Information. Nucleic Acids Res 2000 Jan  1;28(1):10-4   Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Rapp BA,  Wheeler DL (2000). GenBank. Nucleic Acids Res 2000 Jan 1;28(1):15-18.  # noqa: E501

        :param ncbi_taxonomy_id: The ncbi_taxonomy_id of this RcsbEntityHostOrganism.  # noqa: E501
        :type: int
        """

        self._ncbi_taxonomy_id = ncbi_taxonomy_id

    @property
    def pdbx_src_id(self):
        """Gets the pdbx_src_id of this RcsbEntityHostOrganism.  # noqa: E501

        An identifier for an entity segment.  # noqa: E501

        :return: The pdbx_src_id of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: str
        """
        return self._pdbx_src_id

    @pdbx_src_id.setter
    def pdbx_src_id(self, pdbx_src_id):
        """Sets the pdbx_src_id of this RcsbEntityHostOrganism.

        An identifier for an entity segment.  # noqa: E501

        :param pdbx_src_id: The pdbx_src_id of this RcsbEntityHostOrganism.  # noqa: E501
        :type: str
        """
        if pdbx_src_id is None:
            raise ValueError("Invalid value for `pdbx_src_id`, must not be `None`")  # noqa: E501

        self._pdbx_src_id = pdbx_src_id

    @property
    def provenance_source(self):
        """Gets the provenance_source of this RcsbEntityHostOrganism.  # noqa: E501

        A code indicating the provenance of the host organism.  # noqa: E501

        :return: The provenance_source of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: str
        """
        return self._provenance_source

    @provenance_source.setter
    def provenance_source(self, provenance_source):
        """Sets the provenance_source of this RcsbEntityHostOrganism.

        A code indicating the provenance of the host organism.  # noqa: E501

        :param provenance_source: The provenance_source of this RcsbEntityHostOrganism.  # noqa: E501
        :type: str
        """
        allowed_values = ["PDB Primary Data"]  # noqa: E501
        if provenance_source not in allowed_values:
            pass
            #raise ValueError(
            #    "Invalid value for `provenance_source` ({0}), must be one of {1}"  # noqa: E501
            #    .format(provenance_source, allowed_values)
            #)

        self._provenance_source = provenance_source

    @property
    def scientific_name(self):
        """Gets the scientific_name of this RcsbEntityHostOrganism.  # noqa: E501

        The scientific name of the host organism  # noqa: E501

        :return: The scientific_name of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: str
        """
        return self._scientific_name

    @scientific_name.setter
    def scientific_name(self, scientific_name):
        """Sets the scientific_name of this RcsbEntityHostOrganism.

        The scientific name of the host organism  # noqa: E501

        :param scientific_name: The scientific_name of this RcsbEntityHostOrganism.  # noqa: E501
        :type: str
        """

        self._scientific_name = scientific_name

    @property
    def taxonomy_lineage(self):
        """Gets the taxonomy_lineage of this RcsbEntityHostOrganism.  # noqa: E501


        :return: The taxonomy_lineage of this RcsbEntityHostOrganism.  # noqa: E501
        :rtype: list[RcsbEntityHostOrganismTaxonomyLineage]
        """
        return self._taxonomy_lineage

    @taxonomy_lineage.setter
    def taxonomy_lineage(self, taxonomy_lineage):
        """Sets the taxonomy_lineage of this RcsbEntityHostOrganism.


        :param taxonomy_lineage: The taxonomy_lineage of this RcsbEntityHostOrganism.  # noqa: E501
        :type: list[RcsbEntityHostOrganismTaxonomyLineage]
        """

        self._taxonomy_lineage = taxonomy_lineage

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(RcsbEntityHostOrganism, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, RcsbEntityHostOrganism):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
