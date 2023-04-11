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

class DiffrnDetector(object):
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
        'details': 'str',
        'detector': 'str',
        'diffrn_id': 'str',
        'pdbx_collection_date': 'datetime',
        'pdbx_frequency': 'float',
        'type': 'str'
    }

    attribute_map = {
        'details': 'details',
        'detector': 'detector',
        'diffrn_id': 'diffrn_id',
        'pdbx_collection_date': 'pdbx_collection_date',
        'pdbx_frequency': 'pdbx_frequency',
        'type': 'type'
    }

    def __init__(self, details=None, detector=None, diffrn_id=None, pdbx_collection_date=None, pdbx_frequency=None, type=None):  # noqa: E501
        """DiffrnDetector - a model defined in Swagger"""  # noqa: E501
        self._details = None
        self._detector = None
        self._diffrn_id = None
        self._pdbx_collection_date = None
        self._pdbx_frequency = None
        self._type = None
        self.discriminator = None
        if details is not None:
            self.details = details
        if detector is not None:
            self.detector = detector
        self.diffrn_id = diffrn_id
        if pdbx_collection_date is not None:
            self.pdbx_collection_date = pdbx_collection_date
        if pdbx_frequency is not None:
            self.pdbx_frequency = pdbx_frequency
        if type is not None:
            self.type = type

    @property
    def details(self):
        """Gets the details of this DiffrnDetector.  # noqa: E501

        A description of special aspects of the radiation detector.  # noqa: E501

        :return: The details of this DiffrnDetector.  # noqa: E501
        :rtype: str
        """
        return self._details

    @details.setter
    def details(self, details):
        """Sets the details of this DiffrnDetector.

        A description of special aspects of the radiation detector.  # noqa: E501

        :param details: The details of this DiffrnDetector.  # noqa: E501
        :type: str
        """

        self._details = details

    @property
    def detector(self):
        """Gets the detector of this DiffrnDetector.  # noqa: E501

        The general class of the radiation detector.  # noqa: E501

        :return: The detector of this DiffrnDetector.  # noqa: E501
        :rtype: str
        """
        return self._detector

    @detector.setter
    def detector(self, detector):
        """Sets the detector of this DiffrnDetector.

        The general class of the radiation detector.  # noqa: E501

        :param detector: The detector of this DiffrnDetector.  # noqa: E501
        :type: str
        """

        self._detector = detector

    @property
    def diffrn_id(self):
        """Gets the diffrn_id of this DiffrnDetector.  # noqa: E501

        This data item is a pointer to _diffrn.id in the DIFFRN  category.  # noqa: E501

        :return: The diffrn_id of this DiffrnDetector.  # noqa: E501
        :rtype: str
        """
        return self._diffrn_id

    @diffrn_id.setter
    def diffrn_id(self, diffrn_id):
        """Sets the diffrn_id of this DiffrnDetector.

        This data item is a pointer to _diffrn.id in the DIFFRN  category.  # noqa: E501

        :param diffrn_id: The diffrn_id of this DiffrnDetector.  # noqa: E501
        :type: str
        """
        if diffrn_id is None:
            raise ValueError("Invalid value for `diffrn_id`, must not be `None`")  # noqa: E501

        self._diffrn_id = diffrn_id

    @property
    def pdbx_collection_date(self):
        """Gets the pdbx_collection_date of this DiffrnDetector.  # noqa: E501

        The date of data collection.  # noqa: E501

        :return: The pdbx_collection_date of this DiffrnDetector.  # noqa: E501
        :rtype: datetime
        """
        return self._pdbx_collection_date

    @pdbx_collection_date.setter
    def pdbx_collection_date(self, pdbx_collection_date):
        """Sets the pdbx_collection_date of this DiffrnDetector.

        The date of data collection.  # noqa: E501

        :param pdbx_collection_date: The pdbx_collection_date of this DiffrnDetector.  # noqa: E501
        :type: datetime
        """

        self._pdbx_collection_date = pdbx_collection_date

    @property
    def pdbx_frequency(self):
        """Gets the pdbx_frequency of this DiffrnDetector.  # noqa: E501

        The operating frequency of the detector (Hz) used in data collection.  # noqa: E501

        :return: The pdbx_frequency of this DiffrnDetector.  # noqa: E501
        :rtype: float
        """
        return self._pdbx_frequency

    @pdbx_frequency.setter
    def pdbx_frequency(self, pdbx_frequency):
        """Sets the pdbx_frequency of this DiffrnDetector.

        The operating frequency of the detector (Hz) used in data collection.  # noqa: E501

        :param pdbx_frequency: The pdbx_frequency of this DiffrnDetector.  # noqa: E501
        :type: float
        """

        self._pdbx_frequency = pdbx_frequency

    @property
    def type(self):
        """Gets the type of this DiffrnDetector.  # noqa: E501

        The make, model or name of the detector device used.  # noqa: E501

        :return: The type of this DiffrnDetector.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this DiffrnDetector.

        The make, model or name of the detector device used.  # noqa: E501

        :param type: The type of this DiffrnDetector.  # noqa: E501
        :type: str
        """

        self._type = type

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
        if issubclass(DiffrnDetector, dict):
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
        if not isinstance(other, DiffrnDetector):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
