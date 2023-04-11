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

class PdbxEntitySrcSyn(object):
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
        'details': 'str',
        'ncbi_taxonomy_id': 'str',
        'organism_common_name': 'str',
        'organism_scientific': 'str',
        'pdbx_alt_source_flag': 'str',
        'pdbx_beg_seq_num': 'int',
        'pdbx_end_seq_num': 'int',
        'pdbx_src_id': 'int'
    }

    attribute_map = {
        'details': 'details',
        'ncbi_taxonomy_id': 'ncbi_taxonomy_id',
        'organism_common_name': 'organism_common_name',
        'organism_scientific': 'organism_scientific',
        'pdbx_alt_source_flag': 'pdbx_alt_source_flag',
        'pdbx_beg_seq_num': 'pdbx_beg_seq_num',
        'pdbx_end_seq_num': 'pdbx_end_seq_num',
        'pdbx_src_id': 'pdbx_src_id'
    }

    def __init__(self, details=None, ncbi_taxonomy_id=None, organism_common_name=None, organism_scientific=None, pdbx_alt_source_flag=None, pdbx_beg_seq_num=None, pdbx_end_seq_num=None, pdbx_src_id=None):  # noqa: E501
        """PdbxEntitySrcSyn - a model defined in Swagger"""  # noqa: E501
        self._details = None
        self._ncbi_taxonomy_id = None
        self._organism_common_name = None
        self._organism_scientific = None
        self._pdbx_alt_source_flag = None
        self._pdbx_beg_seq_num = None
        self._pdbx_end_seq_num = None
        self._pdbx_src_id = None
        self.discriminator = None
        if details is not None:
            self.details = details
        if ncbi_taxonomy_id is not None:
            self.ncbi_taxonomy_id = ncbi_taxonomy_id
        if organism_common_name is not None:
            self.organism_common_name = organism_common_name
        if organism_scientific is not None:
            self.organism_scientific = organism_scientific
        if pdbx_alt_source_flag is not None:
            self.pdbx_alt_source_flag = pdbx_alt_source_flag
        if pdbx_beg_seq_num is not None:
            self.pdbx_beg_seq_num = pdbx_beg_seq_num
        if pdbx_end_seq_num is not None:
            self.pdbx_end_seq_num = pdbx_end_seq_num
        self.pdbx_src_id = pdbx_src_id

    @property
    def details(self):
        """Gets the details of this PdbxEntitySrcSyn.  # noqa: E501

        A description of special aspects of the source for the  synthetic entity.  # noqa: E501

        :return: The details of this PdbxEntitySrcSyn.  # noqa: E501
        :rtype: str
        """
        return self._details

    @details.setter
    def details(self, details):
        """Sets the details of this PdbxEntitySrcSyn.

        A description of special aspects of the source for the  synthetic entity.  # noqa: E501

        :param details: The details of this PdbxEntitySrcSyn.  # noqa: E501
        :type: str
        """

        self._details = details

    @property
    def ncbi_taxonomy_id(self):
        """Gets the ncbi_taxonomy_id of this PdbxEntitySrcSyn.  # noqa: E501

        NCBI Taxonomy identifier of the organism from which the sequence of  the synthetic entity was derived.   Reference:   Wheeler DL, Chappey C, Lash AE, Leipe DD, Madden TL, Schuler GD,  Tatusova TA, Rapp BA (2000). Database resources of the National  Center for Biotechnology Information. Nucleic Acids Res 2000 Jan  1;28(1):10-4   Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Rapp BA,  Wheeler DL (2000). GenBank. Nucleic Acids Res 2000 Jan 1;28(1):15-18.  # noqa: E501

        :return: The ncbi_taxonomy_id of this PdbxEntitySrcSyn.  # noqa: E501
        :rtype: str
        """
        return self._ncbi_taxonomy_id

    @ncbi_taxonomy_id.setter
    def ncbi_taxonomy_id(self, ncbi_taxonomy_id):
        """Sets the ncbi_taxonomy_id of this PdbxEntitySrcSyn.

        NCBI Taxonomy identifier of the organism from which the sequence of  the synthetic entity was derived.   Reference:   Wheeler DL, Chappey C, Lash AE, Leipe DD, Madden TL, Schuler GD,  Tatusova TA, Rapp BA (2000). Database resources of the National  Center for Biotechnology Information. Nucleic Acids Res 2000 Jan  1;28(1):10-4   Benson DA, Karsch-Mizrachi I, Lipman DJ, Ostell J, Rapp BA,  Wheeler DL (2000). GenBank. Nucleic Acids Res 2000 Jan 1;28(1):15-18.  # noqa: E501

        :param ncbi_taxonomy_id: The ncbi_taxonomy_id of this PdbxEntitySrcSyn.  # noqa: E501
        :type: str
        """

        self._ncbi_taxonomy_id = ncbi_taxonomy_id

    @property
    def organism_common_name(self):
        """Gets the organism_common_name of this PdbxEntitySrcSyn.  # noqa: E501

        The common name of the organism from which the sequence of  the synthetic entity was derived.  # noqa: E501

        :return: The organism_common_name of this PdbxEntitySrcSyn.  # noqa: E501
        :rtype: str
        """
        return self._organism_common_name

    @organism_common_name.setter
    def organism_common_name(self, organism_common_name):
        """Sets the organism_common_name of this PdbxEntitySrcSyn.

        The common name of the organism from which the sequence of  the synthetic entity was derived.  # noqa: E501

        :param organism_common_name: The organism_common_name of this PdbxEntitySrcSyn.  # noqa: E501
        :type: str
        """

        self._organism_common_name = organism_common_name

    @property
    def organism_scientific(self):
        """Gets the organism_scientific of this PdbxEntitySrcSyn.  # noqa: E501

        The scientific name of the organism from which the sequence of  the synthetic entity was derived.  # noqa: E501

        :return: The organism_scientific of this PdbxEntitySrcSyn.  # noqa: E501
        :rtype: str
        """
        return self._organism_scientific

    @organism_scientific.setter
    def organism_scientific(self, organism_scientific):
        """Sets the organism_scientific of this PdbxEntitySrcSyn.

        The scientific name of the organism from which the sequence of  the synthetic entity was derived.  # noqa: E501

        :param organism_scientific: The organism_scientific of this PdbxEntitySrcSyn.  # noqa: E501
        :type: str
        """

        self._organism_scientific = organism_scientific

    @property
    def pdbx_alt_source_flag(self):
        """Gets the pdbx_alt_source_flag of this PdbxEntitySrcSyn.  # noqa: E501

        This data item identifies cases in which an alternative source  modeled.  # noqa: E501

        :return: The pdbx_alt_source_flag of this PdbxEntitySrcSyn.  # noqa: E501
        :rtype: str
        """
        return self._pdbx_alt_source_flag

    @pdbx_alt_source_flag.setter
    def pdbx_alt_source_flag(self, pdbx_alt_source_flag):
        """Sets the pdbx_alt_source_flag of this PdbxEntitySrcSyn.

        This data item identifies cases in which an alternative source  modeled.  # noqa: E501

        :param pdbx_alt_source_flag: The pdbx_alt_source_flag of this PdbxEntitySrcSyn.  # noqa: E501
        :type: str
        """
        allowed_values = ["model", "sample"]  # noqa: E501
        if pdbx_alt_source_flag not in allowed_values:
            raise ValueError(
                "Invalid value for `pdbx_alt_source_flag` ({0}), must be one of {1}"  # noqa: E501
                .format(pdbx_alt_source_flag, allowed_values)
            )

        self._pdbx_alt_source_flag = pdbx_alt_source_flag

    @property
    def pdbx_beg_seq_num(self):
        """Gets the pdbx_beg_seq_num of this PdbxEntitySrcSyn.  # noqa: E501

        The beginning polymer sequence position for the polymer section corresponding  to this source.   A reference to the sequence position in the entity_poly category.  # noqa: E501

        :return: The pdbx_beg_seq_num of this PdbxEntitySrcSyn.  # noqa: E501
        :rtype: int
        """
        return self._pdbx_beg_seq_num

    @pdbx_beg_seq_num.setter
    def pdbx_beg_seq_num(self, pdbx_beg_seq_num):
        """Sets the pdbx_beg_seq_num of this PdbxEntitySrcSyn.

        The beginning polymer sequence position for the polymer section corresponding  to this source.   A reference to the sequence position in the entity_poly category.  # noqa: E501

        :param pdbx_beg_seq_num: The pdbx_beg_seq_num of this PdbxEntitySrcSyn.  # noqa: E501
        :type: int
        """

        self._pdbx_beg_seq_num = pdbx_beg_seq_num

    @property
    def pdbx_end_seq_num(self):
        """Gets the pdbx_end_seq_num of this PdbxEntitySrcSyn.  # noqa: E501

        The ending polymer sequence position for the polymer section corresponding  to this source.   A reference to the sequence position in the entity_poly category.  # noqa: E501

        :return: The pdbx_end_seq_num of this PdbxEntitySrcSyn.  # noqa: E501
        :rtype: int
        """
        return self._pdbx_end_seq_num

    @pdbx_end_seq_num.setter
    def pdbx_end_seq_num(self, pdbx_end_seq_num):
        """Sets the pdbx_end_seq_num of this PdbxEntitySrcSyn.

        The ending polymer sequence position for the polymer section corresponding  to this source.   A reference to the sequence position in the entity_poly category.  # noqa: E501

        :param pdbx_end_seq_num: The pdbx_end_seq_num of this PdbxEntitySrcSyn.  # noqa: E501
        :type: int
        """

        self._pdbx_end_seq_num = pdbx_end_seq_num

    @property
    def pdbx_src_id(self):
        """Gets the pdbx_src_id of this PdbxEntitySrcSyn.  # noqa: E501

        This data item is an ordinal identifier for pdbx_entity_src_syn data records.  # noqa: E501

        :return: The pdbx_src_id of this PdbxEntitySrcSyn.  # noqa: E501
        :rtype: int
        """
        return self._pdbx_src_id

    @pdbx_src_id.setter
    def pdbx_src_id(self, pdbx_src_id):
        """Sets the pdbx_src_id of this PdbxEntitySrcSyn.

        This data item is an ordinal identifier for pdbx_entity_src_syn data records.  # noqa: E501

        :param pdbx_src_id: The pdbx_src_id of this PdbxEntitySrcSyn.  # noqa: E501
        :type: int
        """
        if pdbx_src_id is None:
            raise ValueError("Invalid value for `pdbx_src_id`, must not be `None`")  # noqa: E501

        self._pdbx_src_id = pdbx_src_id

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
        if issubclass(PdbxEntitySrcSyn, dict):
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
        if not isinstance(other, PdbxEntitySrcSyn):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other