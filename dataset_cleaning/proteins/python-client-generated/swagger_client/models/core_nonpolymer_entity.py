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

class CoreNonpolymerEntity(object):
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
        'pdbx_entity_nonpoly': 'PdbxEntityNonpoly',
        'rcsb_nonpolymer_entity': 'RcsbNonpolymerEntity',
        'rcsb_nonpolymer_entity_annotation': 'list[RcsbNonpolymerEntityAnnotation]',
        'rcsb_nonpolymer_entity_container_identifiers': 'RcsbNonpolymerEntityContainerIdentifiers',
        'rcsb_nonpolymer_entity_feature': 'list[RcsbNonpolymerEntityFeature]',
        'rcsb_nonpolymer_entity_feature_summary': 'list[RcsbNonpolymerEntityFeatureSummary]',
        'rcsb_nonpolymer_entity_keywords': 'RcsbNonpolymerEntityKeywords',
        'rcsb_nonpolymer_entity_name_com': 'list[RcsbNonpolymerEntityNameCom]',
        'rcsb_id': 'str',
        'rcsb_latest_revision': 'RcsbLatestRevision'
    }

    attribute_map = {
        'pdbx_entity_nonpoly': 'pdbx_entity_nonpoly',
        'rcsb_nonpolymer_entity': 'rcsb_nonpolymer_entity',
        'rcsb_nonpolymer_entity_annotation': 'rcsb_nonpolymer_entity_annotation',
        'rcsb_nonpolymer_entity_container_identifiers': 'rcsb_nonpolymer_entity_container_identifiers',
        'rcsb_nonpolymer_entity_feature': 'rcsb_nonpolymer_entity_feature',
        'rcsb_nonpolymer_entity_feature_summary': 'rcsb_nonpolymer_entity_feature_summary',
        'rcsb_nonpolymer_entity_keywords': 'rcsb_nonpolymer_entity_keywords',
        'rcsb_nonpolymer_entity_name_com': 'rcsb_nonpolymer_entity_name_com',
        'rcsb_id': 'rcsb_id',
        'rcsb_latest_revision': 'rcsb_latest_revision'
    }

    def __init__(self, pdbx_entity_nonpoly=None, rcsb_nonpolymer_entity=None, rcsb_nonpolymer_entity_annotation=None, rcsb_nonpolymer_entity_container_identifiers=None, rcsb_nonpolymer_entity_feature=None, rcsb_nonpolymer_entity_feature_summary=None, rcsb_nonpolymer_entity_keywords=None, rcsb_nonpolymer_entity_name_com=None, rcsb_id=None, rcsb_latest_revision=None):  # noqa: E501
        """CoreNonpolymerEntity - a model defined in Swagger"""  # noqa: E501
        self._pdbx_entity_nonpoly = None
        self._rcsb_nonpolymer_entity = None
        self._rcsb_nonpolymer_entity_annotation = None
        self._rcsb_nonpolymer_entity_container_identifiers = None
        self._rcsb_nonpolymer_entity_feature = None
        self._rcsb_nonpolymer_entity_feature_summary = None
        self._rcsb_nonpolymer_entity_keywords = None
        self._rcsb_nonpolymer_entity_name_com = None
        self._rcsb_id = None
        self._rcsb_latest_revision = None
        self.discriminator = None
        if pdbx_entity_nonpoly is not None:
            self.pdbx_entity_nonpoly = pdbx_entity_nonpoly
        if rcsb_nonpolymer_entity is not None:
            self.rcsb_nonpolymer_entity = rcsb_nonpolymer_entity
        if rcsb_nonpolymer_entity_annotation is not None:
            self.rcsb_nonpolymer_entity_annotation = rcsb_nonpolymer_entity_annotation
        if rcsb_nonpolymer_entity_container_identifiers is not None:
            self.rcsb_nonpolymer_entity_container_identifiers = rcsb_nonpolymer_entity_container_identifiers
        if rcsb_nonpolymer_entity_feature is not None:
            self.rcsb_nonpolymer_entity_feature = rcsb_nonpolymer_entity_feature
        if rcsb_nonpolymer_entity_feature_summary is not None:
            self.rcsb_nonpolymer_entity_feature_summary = rcsb_nonpolymer_entity_feature_summary
        if rcsb_nonpolymer_entity_keywords is not None:
            self.rcsb_nonpolymer_entity_keywords = rcsb_nonpolymer_entity_keywords
        if rcsb_nonpolymer_entity_name_com is not None:
            self.rcsb_nonpolymer_entity_name_com = rcsb_nonpolymer_entity_name_com
        self.rcsb_id = rcsb_id
        if rcsb_latest_revision is not None:
            self.rcsb_latest_revision = rcsb_latest_revision

    @property
    def pdbx_entity_nonpoly(self):
        """Gets the pdbx_entity_nonpoly of this CoreNonpolymerEntity.  # noqa: E501


        :return: The pdbx_entity_nonpoly of this CoreNonpolymerEntity.  # noqa: E501
        :rtype: PdbxEntityNonpoly
        """
        return self._pdbx_entity_nonpoly

    @pdbx_entity_nonpoly.setter
    def pdbx_entity_nonpoly(self, pdbx_entity_nonpoly):
        """Sets the pdbx_entity_nonpoly of this CoreNonpolymerEntity.


        :param pdbx_entity_nonpoly: The pdbx_entity_nonpoly of this CoreNonpolymerEntity.  # noqa: E501
        :type: PdbxEntityNonpoly
        """

        self._pdbx_entity_nonpoly = pdbx_entity_nonpoly

    @property
    def rcsb_nonpolymer_entity(self):
        """Gets the rcsb_nonpolymer_entity of this CoreNonpolymerEntity.  # noqa: E501


        :return: The rcsb_nonpolymer_entity of this CoreNonpolymerEntity.  # noqa: E501
        :rtype: RcsbNonpolymerEntity
        """
        return self._rcsb_nonpolymer_entity

    @rcsb_nonpolymer_entity.setter
    def rcsb_nonpolymer_entity(self, rcsb_nonpolymer_entity):
        """Sets the rcsb_nonpolymer_entity of this CoreNonpolymerEntity.


        :param rcsb_nonpolymer_entity: The rcsb_nonpolymer_entity of this CoreNonpolymerEntity.  # noqa: E501
        :type: RcsbNonpolymerEntity
        """

        self._rcsb_nonpolymer_entity = rcsb_nonpolymer_entity

    @property
    def rcsb_nonpolymer_entity_annotation(self):
        """Gets the rcsb_nonpolymer_entity_annotation of this CoreNonpolymerEntity.  # noqa: E501


        :return: The rcsb_nonpolymer_entity_annotation of this CoreNonpolymerEntity.  # noqa: E501
        :rtype: list[RcsbNonpolymerEntityAnnotation]
        """
        return self._rcsb_nonpolymer_entity_annotation

    @rcsb_nonpolymer_entity_annotation.setter
    def rcsb_nonpolymer_entity_annotation(self, rcsb_nonpolymer_entity_annotation):
        """Sets the rcsb_nonpolymer_entity_annotation of this CoreNonpolymerEntity.


        :param rcsb_nonpolymer_entity_annotation: The rcsb_nonpolymer_entity_annotation of this CoreNonpolymerEntity.  # noqa: E501
        :type: list[RcsbNonpolymerEntityAnnotation]
        """

        self._rcsb_nonpolymer_entity_annotation = rcsb_nonpolymer_entity_annotation

    @property
    def rcsb_nonpolymer_entity_container_identifiers(self):
        """Gets the rcsb_nonpolymer_entity_container_identifiers of this CoreNonpolymerEntity.  # noqa: E501


        :return: The rcsb_nonpolymer_entity_container_identifiers of this CoreNonpolymerEntity.  # noqa: E501
        :rtype: RcsbNonpolymerEntityContainerIdentifiers
        """
        return self._rcsb_nonpolymer_entity_container_identifiers

    @rcsb_nonpolymer_entity_container_identifiers.setter
    def rcsb_nonpolymer_entity_container_identifiers(self, rcsb_nonpolymer_entity_container_identifiers):
        """Sets the rcsb_nonpolymer_entity_container_identifiers of this CoreNonpolymerEntity.


        :param rcsb_nonpolymer_entity_container_identifiers: The rcsb_nonpolymer_entity_container_identifiers of this CoreNonpolymerEntity.  # noqa: E501
        :type: RcsbNonpolymerEntityContainerIdentifiers
        """

        self._rcsb_nonpolymer_entity_container_identifiers = rcsb_nonpolymer_entity_container_identifiers

    @property
    def rcsb_nonpolymer_entity_feature(self):
        """Gets the rcsb_nonpolymer_entity_feature of this CoreNonpolymerEntity.  # noqa: E501


        :return: The rcsb_nonpolymer_entity_feature of this CoreNonpolymerEntity.  # noqa: E501
        :rtype: list[RcsbNonpolymerEntityFeature]
        """
        return self._rcsb_nonpolymer_entity_feature

    @rcsb_nonpolymer_entity_feature.setter
    def rcsb_nonpolymer_entity_feature(self, rcsb_nonpolymer_entity_feature):
        """Sets the rcsb_nonpolymer_entity_feature of this CoreNonpolymerEntity.


        :param rcsb_nonpolymer_entity_feature: The rcsb_nonpolymer_entity_feature of this CoreNonpolymerEntity.  # noqa: E501
        :type: list[RcsbNonpolymerEntityFeature]
        """

        self._rcsb_nonpolymer_entity_feature = rcsb_nonpolymer_entity_feature

    @property
    def rcsb_nonpolymer_entity_feature_summary(self):
        """Gets the rcsb_nonpolymer_entity_feature_summary of this CoreNonpolymerEntity.  # noqa: E501


        :return: The rcsb_nonpolymer_entity_feature_summary of this CoreNonpolymerEntity.  # noqa: E501
        :rtype: list[RcsbNonpolymerEntityFeatureSummary]
        """
        return self._rcsb_nonpolymer_entity_feature_summary

    @rcsb_nonpolymer_entity_feature_summary.setter
    def rcsb_nonpolymer_entity_feature_summary(self, rcsb_nonpolymer_entity_feature_summary):
        """Sets the rcsb_nonpolymer_entity_feature_summary of this CoreNonpolymerEntity.


        :param rcsb_nonpolymer_entity_feature_summary: The rcsb_nonpolymer_entity_feature_summary of this CoreNonpolymerEntity.  # noqa: E501
        :type: list[RcsbNonpolymerEntityFeatureSummary]
        """

        self._rcsb_nonpolymer_entity_feature_summary = rcsb_nonpolymer_entity_feature_summary

    @property
    def rcsb_nonpolymer_entity_keywords(self):
        """Gets the rcsb_nonpolymer_entity_keywords of this CoreNonpolymerEntity.  # noqa: E501


        :return: The rcsb_nonpolymer_entity_keywords of this CoreNonpolymerEntity.  # noqa: E501
        :rtype: RcsbNonpolymerEntityKeywords
        """
        return self._rcsb_nonpolymer_entity_keywords

    @rcsb_nonpolymer_entity_keywords.setter
    def rcsb_nonpolymer_entity_keywords(self, rcsb_nonpolymer_entity_keywords):
        """Sets the rcsb_nonpolymer_entity_keywords of this CoreNonpolymerEntity.


        :param rcsb_nonpolymer_entity_keywords: The rcsb_nonpolymer_entity_keywords of this CoreNonpolymerEntity.  # noqa: E501
        :type: RcsbNonpolymerEntityKeywords
        """

        self._rcsb_nonpolymer_entity_keywords = rcsb_nonpolymer_entity_keywords

    @property
    def rcsb_nonpolymer_entity_name_com(self):
        """Gets the rcsb_nonpolymer_entity_name_com of this CoreNonpolymerEntity.  # noqa: E501


        :return: The rcsb_nonpolymer_entity_name_com of this CoreNonpolymerEntity.  # noqa: E501
        :rtype: list[RcsbNonpolymerEntityNameCom]
        """
        return self._rcsb_nonpolymer_entity_name_com

    @rcsb_nonpolymer_entity_name_com.setter
    def rcsb_nonpolymer_entity_name_com(self, rcsb_nonpolymer_entity_name_com):
        """Sets the rcsb_nonpolymer_entity_name_com of this CoreNonpolymerEntity.


        :param rcsb_nonpolymer_entity_name_com: The rcsb_nonpolymer_entity_name_com of this CoreNonpolymerEntity.  # noqa: E501
        :type: list[RcsbNonpolymerEntityNameCom]
        """

        self._rcsb_nonpolymer_entity_name_com = rcsb_nonpolymer_entity_name_com

    @property
    def rcsb_id(self):
        """Gets the rcsb_id of this CoreNonpolymerEntity.  # noqa: E501

        A unique identifier for each object in this entity container formed by  an underscore separated concatenation of entry and entity identifiers.  # noqa: E501

        :return: The rcsb_id of this CoreNonpolymerEntity.  # noqa: E501
        :rtype: str
        """
        return self._rcsb_id

    @rcsb_id.setter
    def rcsb_id(self, rcsb_id):
        """Sets the rcsb_id of this CoreNonpolymerEntity.

        A unique identifier for each object in this entity container formed by  an underscore separated concatenation of entry and entity identifiers.  # noqa: E501

        :param rcsb_id: The rcsb_id of this CoreNonpolymerEntity.  # noqa: E501
        :type: str
        """
        if rcsb_id is None:
            raise ValueError("Invalid value for `rcsb_id`, must not be `None`")  # noqa: E501

        self._rcsb_id = rcsb_id

    @property
    def rcsb_latest_revision(self):
        """Gets the rcsb_latest_revision of this CoreNonpolymerEntity.  # noqa: E501


        :return: The rcsb_latest_revision of this CoreNonpolymerEntity.  # noqa: E501
        :rtype: RcsbLatestRevision
        """
        return self._rcsb_latest_revision

    @rcsb_latest_revision.setter
    def rcsb_latest_revision(self, rcsb_latest_revision):
        """Sets the rcsb_latest_revision of this CoreNonpolymerEntity.


        :param rcsb_latest_revision: The rcsb_latest_revision of this CoreNonpolymerEntity.  # noqa: E501
        :type: RcsbLatestRevision
        """

        self._rcsb_latest_revision = rcsb_latest_revision

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
        if issubclass(CoreNonpolymerEntity, dict):
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
        if not isinstance(other, CoreNonpolymerEntity):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
