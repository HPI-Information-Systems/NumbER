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

class RcsbUniprotContainerIdentifiers(object):
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
        'uniprot_id': 'str',
        'reference_sequence_identifiers': 'list[RcsbUniprotContainerIdentifiersReferenceSequenceIdentifiers]'
    }

    attribute_map = {
        'uniprot_id': 'uniprot_id',
        'reference_sequence_identifiers': 'reference_sequence_identifiers'
    }

    def __init__(self, uniprot_id=None, reference_sequence_identifiers=None):  # noqa: E501
        """RcsbUniprotContainerIdentifiers - a model defined in Swagger"""  # noqa: E501
        self._uniprot_id = None
        self._reference_sequence_identifiers = None
        self.discriminator = None
        if uniprot_id is not None:
            self.uniprot_id = uniprot_id
        if reference_sequence_identifiers is not None:
            self.reference_sequence_identifiers = reference_sequence_identifiers

    @property
    def uniprot_id(self):
        """Gets the uniprot_id of this RcsbUniprotContainerIdentifiers.  # noqa: E501

        Primary accession number of a given UniProtKB entry.  # noqa: E501

        :return: The uniprot_id of this RcsbUniprotContainerIdentifiers.  # noqa: E501
        :rtype: str
        """
        return self._uniprot_id

    @uniprot_id.setter
    def uniprot_id(self, uniprot_id):
        """Sets the uniprot_id of this RcsbUniprotContainerIdentifiers.

        Primary accession number of a given UniProtKB entry.  # noqa: E501

        :param uniprot_id: The uniprot_id of this RcsbUniprotContainerIdentifiers.  # noqa: E501
        :type: str
        """

        self._uniprot_id = uniprot_id

    @property
    def reference_sequence_identifiers(self):
        """Gets the reference_sequence_identifiers of this RcsbUniprotContainerIdentifiers.  # noqa: E501


        :return: The reference_sequence_identifiers of this RcsbUniprotContainerIdentifiers.  # noqa: E501
        :rtype: list[RcsbUniprotContainerIdentifiersReferenceSequenceIdentifiers]
        """
        return self._reference_sequence_identifiers

    @reference_sequence_identifiers.setter
    def reference_sequence_identifiers(self, reference_sequence_identifiers):
        """Sets the reference_sequence_identifiers of this RcsbUniprotContainerIdentifiers.


        :param reference_sequence_identifiers: The reference_sequence_identifiers of this RcsbUniprotContainerIdentifiers.  # noqa: E501
        :type: list[RcsbUniprotContainerIdentifiersReferenceSequenceIdentifiers]
        """

        self._reference_sequence_identifiers = reference_sequence_identifiers

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
        if issubclass(RcsbUniprotContainerIdentifiers, dict):
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
        if not isinstance(other, RcsbUniprotContainerIdentifiers):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
