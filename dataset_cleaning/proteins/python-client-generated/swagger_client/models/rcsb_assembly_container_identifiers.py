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

class RcsbAssemblyContainerIdentifiers(object):
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
        'assembly_id': 'str',
        'entry_id': 'str',
        'rcsb_id': 'str',
        'interface_ids': 'list[str]'
    }

    attribute_map = {
        'assembly_id': 'assembly_id',
        'entry_id': 'entry_id',
        'rcsb_id': 'rcsb_id',
        'interface_ids': 'interface_ids'
    }

    def __init__(self, assembly_id=None, entry_id=None, rcsb_id=None, interface_ids=None):  # noqa: E501
        """RcsbAssemblyContainerIdentifiers - a model defined in Swagger"""  # noqa: E501
        self._assembly_id = None
        self._entry_id = None
        self._rcsb_id = None
        self._interface_ids = None
        self.discriminator = None
        self.assembly_id = assembly_id
        self.entry_id = entry_id
        if rcsb_id is not None:
            self.rcsb_id = rcsb_id
        if interface_ids is not None:
            self.interface_ids = interface_ids

    @property
    def assembly_id(self):
        """Gets the assembly_id of this RcsbAssemblyContainerIdentifiers.  # noqa: E501

        Assembly identifier for the container.  # noqa: E501

        :return: The assembly_id of this RcsbAssemblyContainerIdentifiers.  # noqa: E501
        :rtype: str
        """
        return self._assembly_id

    @assembly_id.setter
    def assembly_id(self, assembly_id):
        """Sets the assembly_id of this RcsbAssemblyContainerIdentifiers.

        Assembly identifier for the container.  # noqa: E501

        :param assembly_id: The assembly_id of this RcsbAssemblyContainerIdentifiers.  # noqa: E501
        :type: str
        """
        if assembly_id is None:
            raise ValueError("Invalid value for `assembly_id`, must not be `None`")  # noqa: E501

        self._assembly_id = assembly_id

    @property
    def entry_id(self):
        """Gets the entry_id of this RcsbAssemblyContainerIdentifiers.  # noqa: E501

        Entry identifier for the container.  # noqa: E501

        :return: The entry_id of this RcsbAssemblyContainerIdentifiers.  # noqa: E501
        :rtype: str
        """
        return self._entry_id

    @entry_id.setter
    def entry_id(self, entry_id):
        """Sets the entry_id of this RcsbAssemblyContainerIdentifiers.

        Entry identifier for the container.  # noqa: E501

        :param entry_id: The entry_id of this RcsbAssemblyContainerIdentifiers.  # noqa: E501
        :type: str
        """
        if entry_id is None:
            raise ValueError("Invalid value for `entry_id`, must not be `None`")  # noqa: E501

        self._entry_id = entry_id

    @property
    def rcsb_id(self):
        """Gets the rcsb_id of this RcsbAssemblyContainerIdentifiers.  # noqa: E501

        A unique identifier for each object in this assembly container formed by  a dash separated concatenation of entry and assembly identifiers.  # noqa: E501

        :return: The rcsb_id of this RcsbAssemblyContainerIdentifiers.  # noqa: E501
        :rtype: str
        """
        return self._rcsb_id

    @rcsb_id.setter
    def rcsb_id(self, rcsb_id):
        """Sets the rcsb_id of this RcsbAssemblyContainerIdentifiers.

        A unique identifier for each object in this assembly container formed by  a dash separated concatenation of entry and assembly identifiers.  # noqa: E501

        :param rcsb_id: The rcsb_id of this RcsbAssemblyContainerIdentifiers.  # noqa: E501
        :type: str
        """

        self._rcsb_id = rcsb_id

    @property
    def interface_ids(self):
        """Gets the interface_ids of this RcsbAssemblyContainerIdentifiers.  # noqa: E501


        :return: The interface_ids of this RcsbAssemblyContainerIdentifiers.  # noqa: E501
        :rtype: list[str]
        """
        return self._interface_ids

    @interface_ids.setter
    def interface_ids(self, interface_ids):
        """Sets the interface_ids of this RcsbAssemblyContainerIdentifiers.


        :param interface_ids: The interface_ids of this RcsbAssemblyContainerIdentifiers.  # noqa: E501
        :type: list[str]
        """

        self._interface_ids = interface_ids

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
        if issubclass(RcsbAssemblyContainerIdentifiers, dict):
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
        if not isinstance(other, RcsbAssemblyContainerIdentifiers):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
