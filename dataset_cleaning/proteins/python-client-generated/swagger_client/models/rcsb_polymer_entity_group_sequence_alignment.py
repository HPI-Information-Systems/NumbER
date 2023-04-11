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

class RcsbPolymerEntityGroupSequenceAlignment(object):
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
        'abstract_reference': 'RcsbPolymerEntityGroupSequenceAlignmentAbstractReference',
        'group_members_alignment': 'list[RcsbPolymerEntityGroupSequenceAlignmentGroupMembersAlignment]'
    }

    attribute_map = {
        'abstract_reference': 'abstract_reference',
        'group_members_alignment': 'group_members_alignment'
    }

    def __init__(self, abstract_reference=None, group_members_alignment=None):  # noqa: E501
        """RcsbPolymerEntityGroupSequenceAlignment - a model defined in Swagger"""  # noqa: E501
        self._abstract_reference = None
        self._group_members_alignment = None
        self.discriminator = None
        self.abstract_reference = abstract_reference
        self.group_members_alignment = group_members_alignment

    @property
    def abstract_reference(self):
        """Gets the abstract_reference of this RcsbPolymerEntityGroupSequenceAlignment.  # noqa: E501


        :return: The abstract_reference of this RcsbPolymerEntityGroupSequenceAlignment.  # noqa: E501
        :rtype: RcsbPolymerEntityGroupSequenceAlignmentAbstractReference
        """
        return self._abstract_reference

    @abstract_reference.setter
    def abstract_reference(self, abstract_reference):
        """Sets the abstract_reference of this RcsbPolymerEntityGroupSequenceAlignment.


        :param abstract_reference: The abstract_reference of this RcsbPolymerEntityGroupSequenceAlignment.  # noqa: E501
        :type: RcsbPolymerEntityGroupSequenceAlignmentAbstractReference
        """
        if abstract_reference is None:
            raise ValueError("Invalid value for `abstract_reference`, must not be `None`")  # noqa: E501

        self._abstract_reference = abstract_reference

    @property
    def group_members_alignment(self):
        """Gets the group_members_alignment of this RcsbPolymerEntityGroupSequenceAlignment.  # noqa: E501

        Alignment with a core_entity canonical sequence  # noqa: E501

        :return: The group_members_alignment of this RcsbPolymerEntityGroupSequenceAlignment.  # noqa: E501
        :rtype: list[RcsbPolymerEntityGroupSequenceAlignmentGroupMembersAlignment]
        """
        return self._group_members_alignment

    @group_members_alignment.setter
    def group_members_alignment(self, group_members_alignment):
        """Sets the group_members_alignment of this RcsbPolymerEntityGroupSequenceAlignment.

        Alignment with a core_entity canonical sequence  # noqa: E501

        :param group_members_alignment: The group_members_alignment of this RcsbPolymerEntityGroupSequenceAlignment.  # noqa: E501
        :type: list[RcsbPolymerEntityGroupSequenceAlignmentGroupMembersAlignment]
        """
        if group_members_alignment is None:
            raise ValueError("Invalid value for `group_members_alignment`, must not be `None`")  # noqa: E501

        self._group_members_alignment = group_members_alignment

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
        if issubclass(RcsbPolymerEntityGroupSequenceAlignment, dict):
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
        if not isinstance(other, RcsbPolymerEntityGroupSequenceAlignment):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other