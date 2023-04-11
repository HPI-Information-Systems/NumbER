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

class RcsbChemCompDescriptor(object):
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
        'in_ch_i': 'str',
        'in_ch_i_key': 'str',
        'smiles': 'str',
        'smiles_stereo': 'str',
        'comp_id': 'str'
    }

    attribute_map = {
        'in_ch_i': 'InChI',
        'in_ch_i_key': 'InChIKey',
        'smiles': 'SMILES',
        'smiles_stereo': 'SMILES_stereo',
        'comp_id': 'comp_id'
    }

    def __init__(self, in_ch_i=None, in_ch_i_key=None, smiles=None, smiles_stereo=None, comp_id=None):  # noqa: E501
        """RcsbChemCompDescriptor - a model defined in Swagger"""  # noqa: E501
        self._in_ch_i = None
        self._in_ch_i_key = None
        self._smiles = None
        self._smiles_stereo = None
        self._comp_id = None
        self.discriminator = None
        if in_ch_i is not None:
            self.in_ch_i = in_ch_i
        if in_ch_i_key is not None:
            self.in_ch_i_key = in_ch_i_key
        if smiles is not None:
            self.smiles = smiles
        if smiles_stereo is not None:
            self.smiles_stereo = smiles_stereo
        self.comp_id = comp_id

    @property
    def in_ch_i(self):
        """Gets the in_ch_i of this RcsbChemCompDescriptor.  # noqa: E501

        Standard IUPAC International Chemical Identifier (InChI) descriptor for the chemical component.     InChI, the IUPAC International Chemical Identifier,    by Stephen R Heller, Alan McNaught, Igor Pletnev, Stephen Stein and Dmitrii Tchekhovskoi,    Journal of Cheminformatics, 2015, 7:23;  # noqa: E501

        :return: The in_ch_i of this RcsbChemCompDescriptor.  # noqa: E501
        :rtype: str
        """
        return self._in_ch_i

    @in_ch_i.setter
    def in_ch_i(self, in_ch_i):
        """Sets the in_ch_i of this RcsbChemCompDescriptor.

        Standard IUPAC International Chemical Identifier (InChI) descriptor for the chemical component.     InChI, the IUPAC International Chemical Identifier,    by Stephen R Heller, Alan McNaught, Igor Pletnev, Stephen Stein and Dmitrii Tchekhovskoi,    Journal of Cheminformatics, 2015, 7:23;  # noqa: E501

        :param in_ch_i: The in_ch_i of this RcsbChemCompDescriptor.  # noqa: E501
        :type: str
        """

        self._in_ch_i = in_ch_i

    @property
    def in_ch_i_key(self):
        """Gets the in_ch_i_key of this RcsbChemCompDescriptor.  # noqa: E501

        Standard IUPAC International Chemical Identifier (InChI) descriptor key  for the chemical component   InChI, the IUPAC International Chemical Identifier,  by Stephen R Heller, Alan McNaught, Igor Pletnev, Stephen Stein and Dmitrii Tchekhovskoi,  Journal of Cheminformatics, 2015, 7:23  # noqa: E501

        :return: The in_ch_i_key of this RcsbChemCompDescriptor.  # noqa: E501
        :rtype: str
        """
        return self._in_ch_i_key

    @in_ch_i_key.setter
    def in_ch_i_key(self, in_ch_i_key):
        """Sets the in_ch_i_key of this RcsbChemCompDescriptor.

        Standard IUPAC International Chemical Identifier (InChI) descriptor key  for the chemical component   InChI, the IUPAC International Chemical Identifier,  by Stephen R Heller, Alan McNaught, Igor Pletnev, Stephen Stein and Dmitrii Tchekhovskoi,  Journal of Cheminformatics, 2015, 7:23  # noqa: E501

        :param in_ch_i_key: The in_ch_i_key of this RcsbChemCompDescriptor.  # noqa: E501
        :type: str
        """

        self._in_ch_i_key = in_ch_i_key

    @property
    def smiles(self):
        """Gets the smiles of this RcsbChemCompDescriptor.  # noqa: E501

        Simplified molecular-input line-entry system (SMILES) descriptor for the chemical component.     Weininger D (February 1988). \"SMILES, a chemical language and information system. 1.    Introduction to methodology and encoding rules\". Journal of Chemical Information and Modeling. 28 (1): 31-6.     Weininger D, Weininger A, Weininger JL (May 1989).    \"SMILES. 2. Algorithm for generation of unique SMILES notation\",    Journal of Chemical Information and Modeling. 29 (2): 97-101.  # noqa: E501

        :return: The smiles of this RcsbChemCompDescriptor.  # noqa: E501
        :rtype: str
        """
        return self._smiles

    @smiles.setter
    def smiles(self, smiles):
        """Sets the smiles of this RcsbChemCompDescriptor.

        Simplified molecular-input line-entry system (SMILES) descriptor for the chemical component.     Weininger D (February 1988). \"SMILES, a chemical language and information system. 1.    Introduction to methodology and encoding rules\". Journal of Chemical Information and Modeling. 28 (1): 31-6.     Weininger D, Weininger A, Weininger JL (May 1989).    \"SMILES. 2. Algorithm for generation of unique SMILES notation\",    Journal of Chemical Information and Modeling. 29 (2): 97-101.  # noqa: E501

        :param smiles: The smiles of this RcsbChemCompDescriptor.  # noqa: E501
        :type: str
        """

        self._smiles = smiles

    @property
    def smiles_stereo(self):
        """Gets the smiles_stereo of this RcsbChemCompDescriptor.  # noqa: E501

        Simplified molecular-input line-entry system (SMILES) descriptor for the chemical  component including stereochemical features.   Weininger D (February 1988). \"SMILES, a chemical language and information system. 1.  Introduction to methodology and encoding rules\".  Journal of Chemical Information and Modeling. 28 (1): 31-6.   Weininger D, Weininger A, Weininger JL (May 1989).  \"SMILES. 2. Algorithm for generation of unique SMILES notation\".  Journal of Chemical Information and Modeling. 29 (2): 97-101.  # noqa: E501

        :return: The smiles_stereo of this RcsbChemCompDescriptor.  # noqa: E501
        :rtype: str
        """
        return self._smiles_stereo

    @smiles_stereo.setter
    def smiles_stereo(self, smiles_stereo):
        """Sets the smiles_stereo of this RcsbChemCompDescriptor.

        Simplified molecular-input line-entry system (SMILES) descriptor for the chemical  component including stereochemical features.   Weininger D (February 1988). \"SMILES, a chemical language and information system. 1.  Introduction to methodology and encoding rules\".  Journal of Chemical Information and Modeling. 28 (1): 31-6.   Weininger D, Weininger A, Weininger JL (May 1989).  \"SMILES. 2. Algorithm for generation of unique SMILES notation\".  Journal of Chemical Information and Modeling. 29 (2): 97-101.  # noqa: E501

        :param smiles_stereo: The smiles_stereo of this RcsbChemCompDescriptor.  # noqa: E501
        :type: str
        """

        self._smiles_stereo = smiles_stereo

    @property
    def comp_id(self):
        """Gets the comp_id of this RcsbChemCompDescriptor.  # noqa: E501

        The chemical component identifier.  # noqa: E501

        :return: The comp_id of this RcsbChemCompDescriptor.  # noqa: E501
        :rtype: str
        """
        return self._comp_id

    @comp_id.setter
    def comp_id(self, comp_id):
        """Sets the comp_id of this RcsbChemCompDescriptor.

        The chemical component identifier.  # noqa: E501

        :param comp_id: The comp_id of this RcsbChemCompDescriptor.  # noqa: E501
        :type: str
        """
        if comp_id is None:
            raise ValueError("Invalid value for `comp_id`, must not be `None`")  # noqa: E501

        self._comp_id = comp_id

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
        if issubclass(RcsbChemCompDescriptor, dict):
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
        if not isinstance(other, RcsbChemCompDescriptor):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
