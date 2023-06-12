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

class PdbxNmrSpectrometer(object):
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
        'field_strength': 'float',
        'manufacturer': 'str',
        'model': 'str',
        'spectrometer_id': 'str',
        'type': 'str'
    }

    attribute_map = {
        'details': 'details',
        'field_strength': 'field_strength',
        'manufacturer': 'manufacturer',
        'model': 'model',
        'spectrometer_id': 'spectrometer_id',
        'type': 'type'
    }

    def __init__(self, details=None, field_strength=None, manufacturer=None, model=None, spectrometer_id=None, type=None):  # noqa: E501
        """PdbxNmrSpectrometer - a model defined in Swagger"""  # noqa: E501
        self._details = None
        self._field_strength = None
        self._manufacturer = None
        self._model = None
        self._spectrometer_id = None
        self._type = None
        self.discriminator = None
        if details is not None:
            self.details = details
        if field_strength is not None:
            self.field_strength = field_strength
        if manufacturer is not None:
            self.manufacturer = manufacturer
        if model is not None:
            self.model = model
        self.spectrometer_id = spectrometer_id
        if type is not None:
            self.type = type

    @property
    def details(self):
        """Gets the details of this PdbxNmrSpectrometer.  # noqa: E501

        A text description of the NMR spectrometer.  # noqa: E501

        :return: The details of this PdbxNmrSpectrometer.  # noqa: E501
        :rtype: str
        """
        return self._details

    @details.setter
    def details(self, details):
        """Sets the details of this PdbxNmrSpectrometer.

        A text description of the NMR spectrometer.  # noqa: E501

        :param details: The details of this PdbxNmrSpectrometer.  # noqa: E501
        :type: str
        """

        self._details = details

    @property
    def field_strength(self):
        """Gets the field_strength of this PdbxNmrSpectrometer.  # noqa: E501

        The field strength in MHz of the spectrometer  # noqa: E501

        :return: The field_strength of this PdbxNmrSpectrometer.  # noqa: E501
        :rtype: float
        """
        return self._field_strength

    @field_strength.setter
    def field_strength(self, field_strength):
        """Sets the field_strength of this PdbxNmrSpectrometer.

        The field strength in MHz of the spectrometer  # noqa: E501

        :param field_strength: The field_strength of this PdbxNmrSpectrometer.  # noqa: E501
        :type: float
        """

        self._field_strength = field_strength

    @property
    def manufacturer(self):
        """Gets the manufacturer of this PdbxNmrSpectrometer.  # noqa: E501

        The name of the manufacturer of the spectrometer.  # noqa: E501

        :return: The manufacturer of this PdbxNmrSpectrometer.  # noqa: E501
        :rtype: str
        """
        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, manufacturer):
        """Sets the manufacturer of this PdbxNmrSpectrometer.

        The name of the manufacturer of the spectrometer.  # noqa: E501

        :param manufacturer: The manufacturer of this PdbxNmrSpectrometer.  # noqa: E501
        :type: str
        """

        self._manufacturer = manufacturer

    @property
    def model(self):
        """Gets the model of this PdbxNmrSpectrometer.  # noqa: E501

        The model of the NMR spectrometer.  # noqa: E501

        :return: The model of this PdbxNmrSpectrometer.  # noqa: E501
        :rtype: str
        """
        return self._model

    @model.setter
    def model(self, model):
        """Sets the model of this PdbxNmrSpectrometer.

        The model of the NMR spectrometer.  # noqa: E501

        :param model: The model of this PdbxNmrSpectrometer.  # noqa: E501
        :type: str
        """

        self._model = model

    @property
    def spectrometer_id(self):
        """Gets the spectrometer_id of this PdbxNmrSpectrometer.  # noqa: E501

        Assign a numerical ID to each instrument.  # noqa: E501

        :return: The spectrometer_id of this PdbxNmrSpectrometer.  # noqa: E501
        :rtype: str
        """
        return self._spectrometer_id

    @spectrometer_id.setter
    def spectrometer_id(self, spectrometer_id):
        """Sets the spectrometer_id of this PdbxNmrSpectrometer.

        Assign a numerical ID to each instrument.  # noqa: E501

        :param spectrometer_id: The spectrometer_id of this PdbxNmrSpectrometer.  # noqa: E501
        :type: str
        """
        if spectrometer_id is None:
            raise ValueError("Invalid value for `spectrometer_id`, must not be `None`")  # noqa: E501

        self._spectrometer_id = spectrometer_id

    @property
    def type(self):
        """Gets the type of this PdbxNmrSpectrometer.  # noqa: E501

        Select the instrument manufacturer(s) and the model(s) of the NMR(s) used for this work.  # noqa: E501

        :return: The type of this PdbxNmrSpectrometer.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this PdbxNmrSpectrometer.

        Select the instrument manufacturer(s) and the model(s) of the NMR(s) used for this work.  # noqa: E501

        :param type: The type of this PdbxNmrSpectrometer.  # noqa: E501
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
        if issubclass(PdbxNmrSpectrometer, dict):
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
        if not isinstance(other, PdbxNmrSpectrometer):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other