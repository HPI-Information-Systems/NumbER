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

class RcsbSchemaContainerIdentifiers(object):
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
        'collection_name': 'str',
        'collection_schema_version': 'str',
        'schema_name': 'str'
    }

    attribute_map = {
        'collection_name': 'collection_name',
        'collection_schema_version': 'collection_schema_version',
        'schema_name': 'schema_name'
    }

    def __init__(self, collection_name=None, collection_schema_version=None, schema_name=None):  # noqa: E501
        """RcsbSchemaContainerIdentifiers - a model defined in Swagger"""  # noqa: E501
        self._collection_name = None
        self._collection_schema_version = None
        self._schema_name = None
        self.discriminator = None
        self.collection_name = collection_name
        if collection_schema_version is not None:
            self.collection_schema_version = collection_schema_version
        self.schema_name = schema_name

    @property
    def collection_name(self):
        """Gets the collection_name of this RcsbSchemaContainerIdentifiers.  # noqa: E501

        Collection name associated with the data in the container.  # noqa: E501

        :return: The collection_name of this RcsbSchemaContainerIdentifiers.  # noqa: E501
        :rtype: str
        """
        return self._collection_name

    @collection_name.setter
    def collection_name(self, collection_name):
        """Sets the collection_name of this RcsbSchemaContainerIdentifiers.

        Collection name associated with the data in the container.  # noqa: E501

        :param collection_name: The collection_name of this RcsbSchemaContainerIdentifiers.  # noqa: E501
        :type: str
        """
        if collection_name is None:
            raise ValueError("Invalid value for `collection_name`, must not be `None`")  # noqa: E501

        self._collection_name = collection_name

    @property
    def collection_schema_version(self):
        """Gets the collection_schema_version of this RcsbSchemaContainerIdentifiers.  # noqa: E501

        Version string for the schema and collection.  # noqa: E501

        :return: The collection_schema_version of this RcsbSchemaContainerIdentifiers.  # noqa: E501
        :rtype: str
        """
        return self._collection_schema_version

    @collection_schema_version.setter
    def collection_schema_version(self, collection_schema_version):
        """Sets the collection_schema_version of this RcsbSchemaContainerIdentifiers.

        Version string for the schema and collection.  # noqa: E501

        :param collection_schema_version: The collection_schema_version of this RcsbSchemaContainerIdentifiers.  # noqa: E501
        :type: str
        """

        self._collection_schema_version = collection_schema_version

    @property
    def schema_name(self):
        """Gets the schema_name of this RcsbSchemaContainerIdentifiers.  # noqa: E501

        Schema name associated with the data in the container.  # noqa: E501

        :return: The schema_name of this RcsbSchemaContainerIdentifiers.  # noqa: E501
        :rtype: str
        """
        return self._schema_name

    @schema_name.setter
    def schema_name(self, schema_name):
        """Sets the schema_name of this RcsbSchemaContainerIdentifiers.

        Schema name associated with the data in the container.  # noqa: E501

        :param schema_name: The schema_name of this RcsbSchemaContainerIdentifiers.  # noqa: E501
        :type: str
        """
        if schema_name is None:
            raise ValueError("Invalid value for `schema_name`, must not be `None`")  # noqa: E501

        self._schema_name = schema_name

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
        if issubclass(RcsbSchemaContainerIdentifiers, dict):
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
        if not isinstance(other, RcsbSchemaContainerIdentifiers):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
