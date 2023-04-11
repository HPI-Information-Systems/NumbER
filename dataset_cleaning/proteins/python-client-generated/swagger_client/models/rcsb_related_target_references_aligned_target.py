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

class RcsbRelatedTargetReferencesAlignedTarget(object):
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
        'entity_beg_seq_id': 'int',
        'length': 'int',
        'target_beg_seq_id': 'int'
    }

    attribute_map = {
        'entity_beg_seq_id': 'entity_beg_seq_id',
        'length': 'length',
        'target_beg_seq_id': 'target_beg_seq_id'
    }

    def __init__(self, entity_beg_seq_id=None, length=None, target_beg_seq_id=None):  # noqa: E501
        """RcsbRelatedTargetReferencesAlignedTarget - a model defined in Swagger"""  # noqa: E501
        self._entity_beg_seq_id = None
        self._length = None
        self._target_beg_seq_id = None
        self.discriminator = None
        if entity_beg_seq_id is not None:
            self.entity_beg_seq_id = entity_beg_seq_id
        if length is not None:
            self.length = length
        if target_beg_seq_id is not None:
            self.target_beg_seq_id = target_beg_seq_id

    @property
    def entity_beg_seq_id(self):
        """Gets the entity_beg_seq_id of this RcsbRelatedTargetReferencesAlignedTarget.  # noqa: E501

        The position of the monomer in the entity sequence at which the alignment begins.  # noqa: E501

        :return: The entity_beg_seq_id of this RcsbRelatedTargetReferencesAlignedTarget.  # noqa: E501
        :rtype: int
        """
        return self._entity_beg_seq_id

    @entity_beg_seq_id.setter
    def entity_beg_seq_id(self, entity_beg_seq_id):
        """Sets the entity_beg_seq_id of this RcsbRelatedTargetReferencesAlignedTarget.

        The position of the monomer in the entity sequence at which the alignment begins.  # noqa: E501

        :param entity_beg_seq_id: The entity_beg_seq_id of this RcsbRelatedTargetReferencesAlignedTarget.  # noqa: E501
        :type: int
        """

        self._entity_beg_seq_id = entity_beg_seq_id

    @property
    def length(self):
        """Gets the length of this RcsbRelatedTargetReferencesAlignedTarget.  # noqa: E501

        The length of the alignment.  # noqa: E501

        :return: The length of this RcsbRelatedTargetReferencesAlignedTarget.  # noqa: E501
        :rtype: int
        """
        return self._length

    @length.setter
    def length(self, length):
        """Sets the length of this RcsbRelatedTargetReferencesAlignedTarget.

        The length of the alignment.  # noqa: E501

        :param length: The length of this RcsbRelatedTargetReferencesAlignedTarget.  # noqa: E501
        :type: int
        """

        self._length = length

    @property
    def target_beg_seq_id(self):
        """Gets the target_beg_seq_id of this RcsbRelatedTargetReferencesAlignedTarget.  # noqa: E501

        The position of the monomer in the target sequence at which the alignment begins.  # noqa: E501

        :return: The target_beg_seq_id of this RcsbRelatedTargetReferencesAlignedTarget.  # noqa: E501
        :rtype: int
        """
        return self._target_beg_seq_id

    @target_beg_seq_id.setter
    def target_beg_seq_id(self, target_beg_seq_id):
        """Sets the target_beg_seq_id of this RcsbRelatedTargetReferencesAlignedTarget.

        The position of the monomer in the target sequence at which the alignment begins.  # noqa: E501

        :param target_beg_seq_id: The target_beg_seq_id of this RcsbRelatedTargetReferencesAlignedTarget.  # noqa: E501
        :type: int
        """

        self._target_beg_seq_id = target_beg_seq_id

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
        if issubclass(RcsbRelatedTargetReferencesAlignedTarget, dict):
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
        if not isinstance(other, RcsbRelatedTargetReferencesAlignedTarget):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
