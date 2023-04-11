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

class RcsbPolymerEntityGroupMembership(object):
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
        'group_id': 'str',
        'aggregation_method': 'str',
        'similarity_cutoff': 'float'
    }

    attribute_map = {
        'group_id': 'group_id',
        'aggregation_method': 'aggregation_method',
        'similarity_cutoff': 'similarity_cutoff'
    }

    def __init__(self, group_id=None, aggregation_method=None, similarity_cutoff=None):  # noqa: E501
        """RcsbPolymerEntityGroupMembership - a model defined in Swagger"""  # noqa: E501
        self._group_id = None
        self._aggregation_method = None
        self._similarity_cutoff = None
        self.discriminator = None
        self.group_id = group_id
        self.aggregation_method = aggregation_method
        if similarity_cutoff is not None:
            self.similarity_cutoff = similarity_cutoff

    @property
    def group_id(self):
        """Gets the group_id of this RcsbPolymerEntityGroupMembership.  # noqa: E501

        A unique identifier for a group of entities  # noqa: E501

        :return: The group_id of this RcsbPolymerEntityGroupMembership.  # noqa: E501
        :rtype: str
        """
        return self._group_id

    @group_id.setter
    def group_id(self, group_id):
        """Sets the group_id of this RcsbPolymerEntityGroupMembership.

        A unique identifier for a group of entities  # noqa: E501

        :param group_id: The group_id of this RcsbPolymerEntityGroupMembership.  # noqa: E501
        :type: str
        """
        if group_id is None:
            raise ValueError("Invalid value for `group_id`, must not be `None`")  # noqa: E501

        self._group_id = group_id

    @property
    def aggregation_method(self):
        """Gets the aggregation_method of this RcsbPolymerEntityGroupMembership.  # noqa: E501

        Method used to establish group membership  # noqa: E501

        :return: The aggregation_method of this RcsbPolymerEntityGroupMembership.  # noqa: E501
        :rtype: str
        """
        return self._aggregation_method

    @aggregation_method.setter
    def aggregation_method(self, aggregation_method):
        """Sets the aggregation_method of this RcsbPolymerEntityGroupMembership.

        Method used to establish group membership  # noqa: E501

        :param aggregation_method: The aggregation_method of this RcsbPolymerEntityGroupMembership.  # noqa: E501
        :type: str
        """
        if aggregation_method is None:
            raise ValueError("Invalid value for `aggregation_method`, must not be `None`")  # noqa: E501
        allowed_values = ["sequence_identity", "matching_uniprot_accession"]  # noqa: E501
        if aggregation_method not in allowed_values:
            raise ValueError(
                "Invalid value for `aggregation_method` ({0}), must be one of {1}"  # noqa: E501
                .format(aggregation_method, allowed_values)
            )

        self._aggregation_method = aggregation_method

    @property
    def similarity_cutoff(self):
        """Gets the similarity_cutoff of this RcsbPolymerEntityGroupMembership.  # noqa: E501

        Degree of similarity expressed as a floating-point number  # noqa: E501

        :return: The similarity_cutoff of this RcsbPolymerEntityGroupMembership.  # noqa: E501
        :rtype: float
        """
        return self._similarity_cutoff

    @similarity_cutoff.setter
    def similarity_cutoff(self, similarity_cutoff):
        """Sets the similarity_cutoff of this RcsbPolymerEntityGroupMembership.

        Degree of similarity expressed as a floating-point number  # noqa: E501

        :param similarity_cutoff: The similarity_cutoff of this RcsbPolymerEntityGroupMembership.  # noqa: E501
        :type: float
        """

        self._similarity_cutoff = similarity_cutoff

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
        if issubclass(RcsbPolymerEntityGroupMembership, dict):
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
        if not isinstance(other, RcsbPolymerEntityGroupMembership):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
