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

class RcsbPolymerEntityAlign(object):
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
        'provenance_source': 'str',
        'reference_database_accession': 'str',
        'reference_database_isoform': 'str',
        'reference_database_name': 'str',
        'aligned_regions': 'list[RcsbPolymerEntityAlignAlignedRegions]'
    }

    attribute_map = {
        'provenance_source': 'provenance_source',
        'reference_database_accession': 'reference_database_accession',
        'reference_database_isoform': 'reference_database_isoform',
        'reference_database_name': 'reference_database_name',
        'aligned_regions': 'aligned_regions'
    }

    def __init__(self, provenance_source=None, reference_database_accession=None, reference_database_isoform=None, reference_database_name=None, aligned_regions=None):  # noqa: E501
        """RcsbPolymerEntityAlign - a model defined in Swagger"""  # noqa: E501
        self._provenance_source = None
        self._reference_database_accession = None
        self._reference_database_isoform = None
        self._reference_database_name = None
        self._aligned_regions = None
        self.discriminator = None
        if provenance_source is not None:
            self.provenance_source = provenance_source
        if reference_database_accession is not None:
            self.reference_database_accession = reference_database_accession
        if reference_database_isoform is not None:
            self.reference_database_isoform = reference_database_isoform
        if reference_database_name is not None:
            self.reference_database_name = reference_database_name
        if aligned_regions is not None:
            self.aligned_regions = aligned_regions

    @property
    def provenance_source(self):
        """Gets the provenance_source of this RcsbPolymerEntityAlign.  # noqa: E501

        Code identifying the individual, organization or program that  assigned the reference sequence.  # noqa: E501

        :return: The provenance_source of this RcsbPolymerEntityAlign.  # noqa: E501
        :rtype: str
        """
        return self._provenance_source

    @provenance_source.setter
    def provenance_source(self, provenance_source):
        """Sets the provenance_source of this RcsbPolymerEntityAlign.

        Code identifying the individual, organization or program that  assigned the reference sequence.  # noqa: E501

        :param provenance_source: The provenance_source of this RcsbPolymerEntityAlign.  # noqa: E501
        :type: str
        """

        self._provenance_source = provenance_source

    @property
    def reference_database_accession(self):
        """Gets the reference_database_accession of this RcsbPolymerEntityAlign.  # noqa: E501

        Reference sequence accession code.  # noqa: E501

        :return: The reference_database_accession of this RcsbPolymerEntityAlign.  # noqa: E501
        :rtype: str
        """
        return self._reference_database_accession

    @reference_database_accession.setter
    def reference_database_accession(self, reference_database_accession):
        """Sets the reference_database_accession of this RcsbPolymerEntityAlign.

        Reference sequence accession code.  # noqa: E501

        :param reference_database_accession: The reference_database_accession of this RcsbPolymerEntityAlign.  # noqa: E501
        :type: str
        """

        self._reference_database_accession = reference_database_accession

    @property
    def reference_database_isoform(self):
        """Gets the reference_database_isoform of this RcsbPolymerEntityAlign.  # noqa: E501

        Reference sequence isoform identifier.  # noqa: E501

        :return: The reference_database_isoform of this RcsbPolymerEntityAlign.  # noqa: E501
        :rtype: str
        """
        return self._reference_database_isoform

    @reference_database_isoform.setter
    def reference_database_isoform(self, reference_database_isoform):
        """Sets the reference_database_isoform of this RcsbPolymerEntityAlign.

        Reference sequence isoform identifier.  # noqa: E501

        :param reference_database_isoform: The reference_database_isoform of this RcsbPolymerEntityAlign.  # noqa: E501
        :type: str
        """

        self._reference_database_isoform = reference_database_isoform

    @property
    def reference_database_name(self):
        """Gets the reference_database_name of this RcsbPolymerEntityAlign.  # noqa: E501

        Reference sequence database name.  # noqa: E501

        :return: The reference_database_name of this RcsbPolymerEntityAlign.  # noqa: E501
        :rtype: str
        """
        return self._reference_database_name

    @reference_database_name.setter
    def reference_database_name(self, reference_database_name):
        """Sets the reference_database_name of this RcsbPolymerEntityAlign.

        Reference sequence database name.  # noqa: E501

        :param reference_database_name: The reference_database_name of this RcsbPolymerEntityAlign.  # noqa: E501
        :type: str
        """
        allowed_values = ["EMBL", "GenBank", "NDB", "NORINE", "PDB", "PIR", "PRF", "RefSeq", "UniProt"]  # noqa: E501
        if reference_database_name not in allowed_values:
            raise ValueError(
                "Invalid value for `reference_database_name` ({0}), must be one of {1}"  # noqa: E501
                .format(reference_database_name, allowed_values)
            )

        self._reference_database_name = reference_database_name

    @property
    def aligned_regions(self):
        """Gets the aligned_regions of this RcsbPolymerEntityAlign.  # noqa: E501


        :return: The aligned_regions of this RcsbPolymerEntityAlign.  # noqa: E501
        :rtype: list[RcsbPolymerEntityAlignAlignedRegions]
        """
        return self._aligned_regions

    @aligned_regions.setter
    def aligned_regions(self, aligned_regions):
        """Sets the aligned_regions of this RcsbPolymerEntityAlign.


        :param aligned_regions: The aligned_regions of this RcsbPolymerEntityAlign.  # noqa: E501
        :type: list[RcsbPolymerEntityAlignAlignedRegions]
        """

        self._aligned_regions = aligned_regions

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
        if issubclass(RcsbPolymerEntityAlign, dict):
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
        if not isinstance(other, RcsbPolymerEntityAlign):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
