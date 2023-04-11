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

class RcsbRepositoryHoldingsPrerelease(object):
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
        'entity_id': 'str',
        'seq_one_letter_code': 'str'
    }

    attribute_map = {
        'entity_id': 'entity_id',
        'seq_one_letter_code': 'seq_one_letter_code'
    }

    def __init__(self, entity_id=None, seq_one_letter_code=None):  # noqa: E501
        """RcsbRepositoryHoldingsPrerelease - a model defined in Swagger"""  # noqa: E501
        self._entity_id = None
        self._seq_one_letter_code = None
        self.discriminator = None
        self.entity_id = entity_id
        if seq_one_letter_code is not None:
            self.seq_one_letter_code = seq_one_letter_code

    @property
    def entity_id(self):
        """Gets the entity_id of this RcsbRepositoryHoldingsPrerelease.  # noqa: E501

        The polymer entity identifier.  # noqa: E501

        :return: The entity_id of this RcsbRepositoryHoldingsPrerelease.  # noqa: E501
        :rtype: str
        """
        return self._entity_id

    @entity_id.setter
    def entity_id(self, entity_id):
        """Sets the entity_id of this RcsbRepositoryHoldingsPrerelease.

        The polymer entity identifier.  # noqa: E501

        :param entity_id: The entity_id of this RcsbRepositoryHoldingsPrerelease.  # noqa: E501
        :type: str
        """
        if entity_id is None:
            raise ValueError("Invalid value for `entity_id`, must not be `None`")  # noqa: E501

        self._entity_id = entity_id

    @property
    def seq_one_letter_code(self):
        """Gets the seq_one_letter_code of this RcsbRepositoryHoldingsPrerelease.  # noqa: E501

        Chemical sequence expressed as string of one-letter amino acid codes.  Modifications and non-standard amino acids are represented parenthetically  as 3-letter codes.  # noqa: E501

        :return: The seq_one_letter_code of this RcsbRepositoryHoldingsPrerelease.  # noqa: E501
        :rtype: str
        """
        return self._seq_one_letter_code

    @seq_one_letter_code.setter
    def seq_one_letter_code(self, seq_one_letter_code):
        """Sets the seq_one_letter_code of this RcsbRepositoryHoldingsPrerelease.

        Chemical sequence expressed as string of one-letter amino acid codes.  Modifications and non-standard amino acids are represented parenthetically  as 3-letter codes.  # noqa: E501

        :param seq_one_letter_code: The seq_one_letter_code of this RcsbRepositoryHoldingsPrerelease.  # noqa: E501
        :type: str
        """

        self._seq_one_letter_code = seq_one_letter_code

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
        if issubclass(RcsbRepositoryHoldingsPrerelease, dict):
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
        if not isinstance(other, RcsbRepositoryHoldingsPrerelease):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
