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

class RcsbCompModelProvenance(object):
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
        'entry_id': 'str',
        'source_db': 'str',
        'source_filename': 'str',
        'source_pae_url': 'str',
        'source_url': 'str'
    }

    attribute_map = {
        'entry_id': 'entry_id',
        'source_db': 'source_db',
        'source_filename': 'source_filename',
        'source_pae_url': 'source_pae_url',
        'source_url': 'source_url'
    }

    def __init__(self, entry_id=None, source_db=None, source_filename=None, source_pae_url=None, source_url=None):  # noqa: E501
        """RcsbCompModelProvenance - a model defined in Swagger"""  # noqa: E501
        self._entry_id = None
        self._source_db = None
        self._source_filename = None
        self._source_pae_url = None
        self._source_url = None
        self.discriminator = None
        self.entry_id = entry_id
        if source_db is not None:
            self.source_db = source_db
        if source_filename is not None:
            self.source_filename = source_filename
        if source_pae_url is not None:
            self.source_pae_url = source_pae_url
        if source_url is not None:
            self.source_url = source_url

    @property
    def entry_id(self):
        """Gets the entry_id of this RcsbCompModelProvenance.  # noqa: E501

        Entry identifier corresponding to the computed structure model.  # noqa: E501

        :return: The entry_id of this RcsbCompModelProvenance.  # noqa: E501
        :rtype: str
        """
        return self._entry_id

    @entry_id.setter
    def entry_id(self, entry_id):
        """Sets the entry_id of this RcsbCompModelProvenance.

        Entry identifier corresponding to the computed structure model.  # noqa: E501

        :param entry_id: The entry_id of this RcsbCompModelProvenance.  # noqa: E501
        :type: str
        """
        if entry_id is None:
            raise ValueError("Invalid value for `entry_id`, must not be `None`")  # noqa: E501

        self._entry_id = entry_id

    @property
    def source_db(self):
        """Gets the source_db of this RcsbCompModelProvenance.  # noqa: E501

        Source database for the computed structure model.  # noqa: E501

        :return: The source_db of this RcsbCompModelProvenance.  # noqa: E501
        :rtype: str
        """
        return self._source_db

    @source_db.setter
    def source_db(self, source_db):
        """Sets the source_db of this RcsbCompModelProvenance.

        Source database for the computed structure model.  # noqa: E501

        :param source_db: The source_db of this RcsbCompModelProvenance.  # noqa: E501
        :type: str
        """
        allowed_values = ["AlphaFoldDB", "ModelArchive"]  # noqa: E501
        if source_db not in allowed_values:
            raise ValueError(
                "Invalid value for `source_db` ({0}), must be one of {1}"  # noqa: E501
                .format(source_db, allowed_values)
            )

        self._source_db = source_db

    @property
    def source_filename(self):
        """Gets the source_filename of this RcsbCompModelProvenance.  # noqa: E501

        Source filename for the computed structure model.  # noqa: E501

        :return: The source_filename of this RcsbCompModelProvenance.  # noqa: E501
        :rtype: str
        """
        return self._source_filename

    @source_filename.setter
    def source_filename(self, source_filename):
        """Sets the source_filename of this RcsbCompModelProvenance.

        Source filename for the computed structure model.  # noqa: E501

        :param source_filename: The source_filename of this RcsbCompModelProvenance.  # noqa: E501
        :type: str
        """

        self._source_filename = source_filename

    @property
    def source_pae_url(self):
        """Gets the source_pae_url of this RcsbCompModelProvenance.  # noqa: E501

        Source URL for computed structure model predicted aligned error (PAE) json file.  # noqa: E501

        :return: The source_pae_url of this RcsbCompModelProvenance.  # noqa: E501
        :rtype: str
        """
        return self._source_pae_url

    @source_pae_url.setter
    def source_pae_url(self, source_pae_url):
        """Sets the source_pae_url of this RcsbCompModelProvenance.

        Source URL for computed structure model predicted aligned error (PAE) json file.  # noqa: E501

        :param source_pae_url: The source_pae_url of this RcsbCompModelProvenance.  # noqa: E501
        :type: str
        """

        self._source_pae_url = source_pae_url

    @property
    def source_url(self):
        """Gets the source_url of this RcsbCompModelProvenance.  # noqa: E501

        Source URL for computed structure model file.  # noqa: E501

        :return: The source_url of this RcsbCompModelProvenance.  # noqa: E501
        :rtype: str
        """
        return self._source_url

    @source_url.setter
    def source_url(self, source_url):
        """Sets the source_url of this RcsbCompModelProvenance.

        Source URL for computed structure model file.  # noqa: E501

        :param source_url: The source_url of this RcsbCompModelProvenance.  # noqa: E501
        :type: str
        """

        self._source_url = source_url

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
        if issubclass(RcsbCompModelProvenance, dict):
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
        if not isinstance(other, RcsbCompModelProvenance):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other