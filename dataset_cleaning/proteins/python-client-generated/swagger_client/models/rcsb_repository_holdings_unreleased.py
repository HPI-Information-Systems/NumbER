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

class RcsbRepositoryHoldingsUnreleased(object):
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
        'audit_authors': 'list[str]',
        'author_prerelease_sequence_status': 'str',
        'deposit_date': 'datetime',
        'deposit_date_coordinates': 'datetime',
        'deposit_date_nmr_restraints': 'datetime',
        'deposit_date_structure_factors': 'datetime',
        'hold_date_coordinates': 'datetime',
        'hold_date_nmr_restraints': 'datetime',
        'hold_date_structure_factors': 'datetime',
        'prerelease_sequence_available_flag': 'str',
        'release_date': 'datetime',
        'status_code': 'str',
        'title': 'str'
    }

    attribute_map = {
        'audit_authors': 'audit_authors',
        'author_prerelease_sequence_status': 'author_prerelease_sequence_status',
        'deposit_date': 'deposit_date',
        'deposit_date_coordinates': 'deposit_date_coordinates',
        'deposit_date_nmr_restraints': 'deposit_date_nmr_restraints',
        'deposit_date_structure_factors': 'deposit_date_structure_factors',
        'hold_date_coordinates': 'hold_date_coordinates',
        'hold_date_nmr_restraints': 'hold_date_nmr_restraints',
        'hold_date_structure_factors': 'hold_date_structure_factors',
        'prerelease_sequence_available_flag': 'prerelease_sequence_available_flag',
        'release_date': 'release_date',
        'status_code': 'status_code',
        'title': 'title'
    }

    def __init__(self, audit_authors=None, author_prerelease_sequence_status=None, deposit_date=None, deposit_date_coordinates=None, deposit_date_nmr_restraints=None, deposit_date_structure_factors=None, hold_date_coordinates=None, hold_date_nmr_restraints=None, hold_date_structure_factors=None, prerelease_sequence_available_flag=None, release_date=None, status_code=None, title=None):  # noqa: E501
        """RcsbRepositoryHoldingsUnreleased - a model defined in Swagger"""  # noqa: E501
        self._audit_authors = None
        self._author_prerelease_sequence_status = None
        self._deposit_date = None
        self._deposit_date_coordinates = None
        self._deposit_date_nmr_restraints = None
        self._deposit_date_structure_factors = None
        self._hold_date_coordinates = None
        self._hold_date_nmr_restraints = None
        self._hold_date_structure_factors = None
        self._prerelease_sequence_available_flag = None
        self._release_date = None
        self._status_code = None
        self._title = None
        self.discriminator = None
        if audit_authors is not None:
            self.audit_authors = audit_authors
        if author_prerelease_sequence_status is not None:
            self.author_prerelease_sequence_status = author_prerelease_sequence_status
        if deposit_date is not None:
            self.deposit_date = deposit_date
        if deposit_date_coordinates is not None:
            self.deposit_date_coordinates = deposit_date_coordinates
        if deposit_date_nmr_restraints is not None:
            self.deposit_date_nmr_restraints = deposit_date_nmr_restraints
        if deposit_date_structure_factors is not None:
            self.deposit_date_structure_factors = deposit_date_structure_factors
        if hold_date_coordinates is not None:
            self.hold_date_coordinates = hold_date_coordinates
        if hold_date_nmr_restraints is not None:
            self.hold_date_nmr_restraints = hold_date_nmr_restraints
        if hold_date_structure_factors is not None:
            self.hold_date_structure_factors = hold_date_structure_factors
        if prerelease_sequence_available_flag is not None:
            self.prerelease_sequence_available_flag = prerelease_sequence_available_flag
        if release_date is not None:
            self.release_date = release_date
        if status_code is not None:
            self.status_code = status_code
        if title is not None:
            self.title = title

    @property
    def audit_authors(self):
        """Gets the audit_authors of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        Names of the authors of the entry.  # noqa: E501

        :return: The audit_authors of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: list[str]
        """
        return self._audit_authors

    @audit_authors.setter
    def audit_authors(self, audit_authors):
        """Sets the audit_authors of this RcsbRepositoryHoldingsUnreleased.

        Names of the authors of the entry.  # noqa: E501

        :param audit_authors: The audit_authors of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: list[str]
        """

        self._audit_authors = audit_authors

    @property
    def author_prerelease_sequence_status(self):
        """Gets the author_prerelease_sequence_status of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The prerelease status for the molecular sequences for the entry .  # noqa: E501

        :return: The author_prerelease_sequence_status of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: str
        """
        return self._author_prerelease_sequence_status

    @author_prerelease_sequence_status.setter
    def author_prerelease_sequence_status(self, author_prerelease_sequence_status):
        """Sets the author_prerelease_sequence_status of this RcsbRepositoryHoldingsUnreleased.

        The prerelease status for the molecular sequences for the entry .  # noqa: E501

        :param author_prerelease_sequence_status: The author_prerelease_sequence_status of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: str
        """
        allowed_values = ["HOLD FOR PUBLICATION", "HOLD FOR RELEASE", "RELEASE IMMEDIATELY", "RELEASE NOW"]  # noqa: E501
        if author_prerelease_sequence_status not in allowed_values:
            raise ValueError(
                "Invalid value for `author_prerelease_sequence_status` ({0}), must be one of {1}"  # noqa: E501
                .format(author_prerelease_sequence_status, allowed_values)
            )

        self._author_prerelease_sequence_status = author_prerelease_sequence_status

    @property
    def deposit_date(self):
        """Gets the deposit_date of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The entry deposition date.  # noqa: E501

        :return: The deposit_date of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: datetime
        """
        return self._deposit_date

    @deposit_date.setter
    def deposit_date(self, deposit_date):
        """Sets the deposit_date of this RcsbRepositoryHoldingsUnreleased.

        The entry deposition date.  # noqa: E501

        :param deposit_date: The deposit_date of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: datetime
        """

        self._deposit_date = deposit_date

    @property
    def deposit_date_coordinates(self):
        """Gets the deposit_date_coordinates of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The deposition date of entry coordinate data.  # noqa: E501

        :return: The deposit_date_coordinates of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: datetime
        """
        return self._deposit_date_coordinates

    @deposit_date_coordinates.setter
    def deposit_date_coordinates(self, deposit_date_coordinates):
        """Sets the deposit_date_coordinates of this RcsbRepositoryHoldingsUnreleased.

        The deposition date of entry coordinate data.  # noqa: E501

        :param deposit_date_coordinates: The deposit_date_coordinates of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: datetime
        """

        self._deposit_date_coordinates = deposit_date_coordinates

    @property
    def deposit_date_nmr_restraints(self):
        """Gets the deposit_date_nmr_restraints of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The deposition date of entry NMR restraint data.  # noqa: E501

        :return: The deposit_date_nmr_restraints of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: datetime
        """
        return self._deposit_date_nmr_restraints

    @deposit_date_nmr_restraints.setter
    def deposit_date_nmr_restraints(self, deposit_date_nmr_restraints):
        """Sets the deposit_date_nmr_restraints of this RcsbRepositoryHoldingsUnreleased.

        The deposition date of entry NMR restraint data.  # noqa: E501

        :param deposit_date_nmr_restraints: The deposit_date_nmr_restraints of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: datetime
        """

        self._deposit_date_nmr_restraints = deposit_date_nmr_restraints

    @property
    def deposit_date_structure_factors(self):
        """Gets the deposit_date_structure_factors of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The deposition date of entry structure factor data.  # noqa: E501

        :return: The deposit_date_structure_factors of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: datetime
        """
        return self._deposit_date_structure_factors

    @deposit_date_structure_factors.setter
    def deposit_date_structure_factors(self, deposit_date_structure_factors):
        """Sets the deposit_date_structure_factors of this RcsbRepositoryHoldingsUnreleased.

        The deposition date of entry structure factor data.  # noqa: E501

        :param deposit_date_structure_factors: The deposit_date_structure_factors of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: datetime
        """

        self._deposit_date_structure_factors = deposit_date_structure_factors

    @property
    def hold_date_coordinates(self):
        """Gets the hold_date_coordinates of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The embargo date of entry coordinate data.  # noqa: E501

        :return: The hold_date_coordinates of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: datetime
        """
        return self._hold_date_coordinates

    @hold_date_coordinates.setter
    def hold_date_coordinates(self, hold_date_coordinates):
        """Sets the hold_date_coordinates of this RcsbRepositoryHoldingsUnreleased.

        The embargo date of entry coordinate data.  # noqa: E501

        :param hold_date_coordinates: The hold_date_coordinates of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: datetime
        """

        self._hold_date_coordinates = hold_date_coordinates

    @property
    def hold_date_nmr_restraints(self):
        """Gets the hold_date_nmr_restraints of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The embargo date of entry NMR restraint data.  # noqa: E501

        :return: The hold_date_nmr_restraints of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: datetime
        """
        return self._hold_date_nmr_restraints

    @hold_date_nmr_restraints.setter
    def hold_date_nmr_restraints(self, hold_date_nmr_restraints):
        """Sets the hold_date_nmr_restraints of this RcsbRepositoryHoldingsUnreleased.

        The embargo date of entry NMR restraint data.  # noqa: E501

        :param hold_date_nmr_restraints: The hold_date_nmr_restraints of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: datetime
        """

        self._hold_date_nmr_restraints = hold_date_nmr_restraints

    @property
    def hold_date_structure_factors(self):
        """Gets the hold_date_structure_factors of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The embargo date of entry structure factor data.  # noqa: E501

        :return: The hold_date_structure_factors of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: datetime
        """
        return self._hold_date_structure_factors

    @hold_date_structure_factors.setter
    def hold_date_structure_factors(self, hold_date_structure_factors):
        """Sets the hold_date_structure_factors of this RcsbRepositoryHoldingsUnreleased.

        The embargo date of entry structure factor data.  # noqa: E501

        :param hold_date_structure_factors: The hold_date_structure_factors of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: datetime
        """

        self._hold_date_structure_factors = hold_date_structure_factors

    @property
    def prerelease_sequence_available_flag(self):
        """Gets the prerelease_sequence_available_flag of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        A flag to indicate the molecular sequences for an unreleased entry are available.  # noqa: E501

        :return: The prerelease_sequence_available_flag of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: str
        """
        return self._prerelease_sequence_available_flag

    @prerelease_sequence_available_flag.setter
    def prerelease_sequence_available_flag(self, prerelease_sequence_available_flag):
        """Sets the prerelease_sequence_available_flag of this RcsbRepositoryHoldingsUnreleased.

        A flag to indicate the molecular sequences for an unreleased entry are available.  # noqa: E501

        :param prerelease_sequence_available_flag: The prerelease_sequence_available_flag of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: str
        """
        allowed_values = ["N", "Y"]  # noqa: E501
        if prerelease_sequence_available_flag not in allowed_values:
            raise ValueError(
                "Invalid value for `prerelease_sequence_available_flag` ({0}), must be one of {1}"  # noqa: E501
                .format(prerelease_sequence_available_flag, allowed_values)
            )

        self._prerelease_sequence_available_flag = prerelease_sequence_available_flag

    @property
    def release_date(self):
        """Gets the release_date of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The release date for the entry.  # noqa: E501

        :return: The release_date of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: datetime
        """
        return self._release_date

    @release_date.setter
    def release_date(self, release_date):
        """Sets the release_date of this RcsbRepositoryHoldingsUnreleased.

        The release date for the entry.  # noqa: E501

        :param release_date: The release_date of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: datetime
        """

        self._release_date = release_date

    @property
    def status_code(self):
        """Gets the status_code of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The release status for the entry.  # noqa: E501

        :return: The status_code of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: str
        """
        return self._status_code

    @status_code.setter
    def status_code(self, status_code):
        """Sets the status_code of this RcsbRepositoryHoldingsUnreleased.

        The release status for the entry.  # noqa: E501

        :param status_code: The status_code of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: str
        """
        allowed_values = ["AUCO", "AUTH", "HOLD", "HPUB", "POLC", "PROC", "REFI", "REPL", "WAIT", "WDRN"]  # noqa: E501
        if status_code not in allowed_values:
            raise ValueError(
                "Invalid value for `status_code` ({0}), must be one of {1}"  # noqa: E501
                .format(status_code, allowed_values)
            )

        self._status_code = status_code

    @property
    def title(self):
        """Gets the title of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501

        The entry title.  # noqa: E501

        :return: The title of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :rtype: str
        """
        return self._title

    @title.setter
    def title(self, title):
        """Sets the title of this RcsbRepositoryHoldingsUnreleased.

        The entry title.  # noqa: E501

        :param title: The title of this RcsbRepositoryHoldingsUnreleased.  # noqa: E501
        :type: str
        """

        self._title = title

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
        if issubclass(RcsbRepositoryHoldingsUnreleased, dict):
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
        if not isinstance(other, RcsbRepositoryHoldingsUnreleased):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other