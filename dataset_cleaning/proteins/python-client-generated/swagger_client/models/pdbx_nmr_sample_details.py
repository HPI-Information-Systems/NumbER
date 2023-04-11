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

class PdbxNmrSampleDetails(object):
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
        'contents': 'str',
        'details': 'str',
        'label': 'str',
        'solution_id': 'str',
        'solvent_system': 'str',
        'type': 'str'
    }

    attribute_map = {
        'contents': 'contents',
        'details': 'details',
        'label': 'label',
        'solution_id': 'solution_id',
        'solvent_system': 'solvent_system',
        'type': 'type'
    }

    def __init__(self, contents=None, details=None, label=None, solution_id=None, solvent_system=None, type=None):  # noqa: E501
        """PdbxNmrSampleDetails - a model defined in Swagger"""  # noqa: E501
        self._contents = None
        self._details = None
        self._label = None
        self._solution_id = None
        self._solvent_system = None
        self._type = None
        self.discriminator = None
        if contents is not None:
            self.contents = contents
        if details is not None:
            self.details = details
        if label is not None:
            self.label = label
        self.solution_id = solution_id
        if solvent_system is not None:
            self.solvent_system = solvent_system
        if type is not None:
            self.type = type

    @property
    def contents(self):
        """Gets the contents of this PdbxNmrSampleDetails.  # noqa: E501

        A complete description of each NMR sample. Include the concentration and concentration units for each component (include buffers, etc.). For each component describe the isotopic composition, including the % labeling level, if known.  For example: 1. Uniform (random) labeling with 15N: U-15N 2. Uniform (random) labeling with 13C, 15N at known labeling    levels: U-95% 13C;U-98% 15N 3. Residue selective labeling: U-95% 15N-Thymine 4. Site specific labeling: 95% 13C-Ala18, 5. Natural abundance labeling in an otherwise uniformly labeled    biomolecule is designated by NA: U-13C; NA-K,H  # noqa: E501

        :return: The contents of this PdbxNmrSampleDetails.  # noqa: E501
        :rtype: str
        """
        return self._contents

    @contents.setter
    def contents(self, contents):
        """Sets the contents of this PdbxNmrSampleDetails.

        A complete description of each NMR sample. Include the concentration and concentration units for each component (include buffers, etc.). For each component describe the isotopic composition, including the % labeling level, if known.  For example: 1. Uniform (random) labeling with 15N: U-15N 2. Uniform (random) labeling with 13C, 15N at known labeling    levels: U-95% 13C;U-98% 15N 3. Residue selective labeling: U-95% 15N-Thymine 4. Site specific labeling: 95% 13C-Ala18, 5. Natural abundance labeling in an otherwise uniformly labeled    biomolecule is designated by NA: U-13C; NA-K,H  # noqa: E501

        :param contents: The contents of this PdbxNmrSampleDetails.  # noqa: E501
        :type: str
        """

        self._contents = contents

    @property
    def details(self):
        """Gets the details of this PdbxNmrSampleDetails.  # noqa: E501

        Brief description of the sample providing additional information not captured by other items in the category.  # noqa: E501

        :return: The details of this PdbxNmrSampleDetails.  # noqa: E501
        :rtype: str
        """
        return self._details

    @details.setter
    def details(self, details):
        """Sets the details of this PdbxNmrSampleDetails.

        Brief description of the sample providing additional information not captured by other items in the category.  # noqa: E501

        :param details: The details of this PdbxNmrSampleDetails.  # noqa: E501
        :type: str
        """

        self._details = details

    @property
    def label(self):
        """Gets the label of this PdbxNmrSampleDetails.  # noqa: E501

        A value that uniquely identifies this sample from the other samples listed in the entry.  # noqa: E501

        :return: The label of this PdbxNmrSampleDetails.  # noqa: E501
        :rtype: str
        """
        return self._label

    @label.setter
    def label(self, label):
        """Sets the label of this PdbxNmrSampleDetails.

        A value that uniquely identifies this sample from the other samples listed in the entry.  # noqa: E501

        :param label: The label of this PdbxNmrSampleDetails.  # noqa: E501
        :type: str
        """

        self._label = label

    @property
    def solution_id(self):
        """Gets the solution_id of this PdbxNmrSampleDetails.  # noqa: E501

        The name (number) of the sample.  # noqa: E501

        :return: The solution_id of this PdbxNmrSampleDetails.  # noqa: E501
        :rtype: str
        """
        return self._solution_id

    @solution_id.setter
    def solution_id(self, solution_id):
        """Sets the solution_id of this PdbxNmrSampleDetails.

        The name (number) of the sample.  # noqa: E501

        :param solution_id: The solution_id of this PdbxNmrSampleDetails.  # noqa: E501
        :type: str
        """
        if solution_id is None:
            raise ValueError("Invalid value for `solution_id`, must not be `None`")  # noqa: E501

        self._solution_id = solution_id

    @property
    def solvent_system(self):
        """Gets the solvent_system of this PdbxNmrSampleDetails.  # noqa: E501

        The solvent system used for this sample.  # noqa: E501

        :return: The solvent_system of this PdbxNmrSampleDetails.  # noqa: E501
        :rtype: str
        """
        return self._solvent_system

    @solvent_system.setter
    def solvent_system(self, solvent_system):
        """Sets the solvent_system of this PdbxNmrSampleDetails.

        The solvent system used for this sample.  # noqa: E501

        :param solvent_system: The solvent_system of this PdbxNmrSampleDetails.  # noqa: E501
        :type: str
        """

        self._solvent_system = solvent_system

    @property
    def type(self):
        """Gets the type of this PdbxNmrSampleDetails.  # noqa: E501

        A descriptive term for the sample that defines the general physical properties of the sample.  # noqa: E501

        :return: The type of this PdbxNmrSampleDetails.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this PdbxNmrSampleDetails.

        A descriptive term for the sample that defines the general physical properties of the sample.  # noqa: E501

        :param type: The type of this PdbxNmrSampleDetails.  # noqa: E501
        :type: str
        """
        allowed_values = ["bicelle", "emulsion", "fiber", "fibrous protein", "filamentous virus", "gel solid", "gel solution", "liposome", "lyophilized powder", "membrane", "micelle", "oriented membrane film", "polycrystalline powder", "reverse micelle", "single crystal", "solid", "solution"]  # noqa: E501
        if type not in allowed_values:
            raise ValueError(
                "Invalid value for `type` ({0}), must be one of {1}"  # noqa: E501
                .format(type, allowed_values)
            )

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
        if issubclass(PdbxNmrSampleDetails, dict):
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
        if not isinstance(other, PdbxNmrSampleDetails):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other