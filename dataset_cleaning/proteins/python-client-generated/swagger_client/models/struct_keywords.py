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

class StructKeywords(object):
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
        'pdbx_keywords': 'str',
        'text': 'str'
    }

    attribute_map = {
        'pdbx_keywords': 'pdbx_keywords',
        'text': 'text'
    }

    def __init__(self, pdbx_keywords=None, text=None):  # noqa: E501
        """StructKeywords - a model defined in Swagger"""  # noqa: E501
        self._pdbx_keywords = None
        self._text = None
        self.discriminator = None
        if pdbx_keywords is not None:
            self.pdbx_keywords = pdbx_keywords
        if text is not None:
            self.text = text

    @property
    def pdbx_keywords(self):
        """Gets the pdbx_keywords of this StructKeywords.  # noqa: E501

        Terms characterizing the macromolecular structure.  # noqa: E501

        :return: The pdbx_keywords of this StructKeywords.  # noqa: E501
        :rtype: str
        """
        return self._pdbx_keywords

    @pdbx_keywords.setter
    def pdbx_keywords(self, pdbx_keywords):
        """Sets the pdbx_keywords of this StructKeywords.

        Terms characterizing the macromolecular structure.  # noqa: E501

        :param pdbx_keywords: The pdbx_keywords of this StructKeywords.  # noqa: E501
        :type: str
        """

        self._pdbx_keywords = pdbx_keywords

    @property
    def text(self):
        """Gets the text of this StructKeywords.  # noqa: E501

        Keywords describing this structure.  # noqa: E501

        :return: The text of this StructKeywords.  # noqa: E501
        :rtype: str
        """
        return self._text

    @text.setter
    def text(self, text):
        """Sets the text of this StructKeywords.

        Keywords describing this structure.  # noqa: E501

        :param text: The text of this StructKeywords.  # noqa: E501
        :type: str
        """

        self._text = text

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
        if issubclass(StructKeywords, dict):
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
        if not isinstance(other, StructKeywords):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
