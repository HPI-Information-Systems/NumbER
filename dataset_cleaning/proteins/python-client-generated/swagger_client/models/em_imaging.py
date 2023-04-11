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

class EmImaging(object):
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
        'accelerating_voltage': 'int',
        'alignment_procedure': 'str',
        'astigmatism': 'str',
        'c2_aperture_diameter': 'float',
        'calibrated_defocus_max': 'float',
        'calibrated_defocus_min': 'float',
        'calibrated_magnification': 'int',
        'cryogen': 'str',
        '_date': 'datetime',
        'details': 'str',
        'detector_distance': 'float',
        'electron_beam_tilt_params': 'str',
        'electron_source': 'str',
        'id': 'str',
        'illumination_mode': 'str',
        'microscope_model': 'str',
        'mode': 'str',
        'nominal_cs': 'float',
        'nominal_defocus_max': 'float',
        'nominal_defocus_min': 'float',
        'nominal_magnification': 'int',
        'recording_temperature_maximum': 'float',
        'recording_temperature_minimum': 'float',
        'residual_tilt': 'float',
        'specimen_holder_model': 'str',
        'specimen_holder_type': 'str',
        'specimen_id': 'str',
        'temperature': 'float',
        'tilt_angle_max': 'float',
        'tilt_angle_min': 'float'
    }

    attribute_map = {
        'accelerating_voltage': 'accelerating_voltage',
        'alignment_procedure': 'alignment_procedure',
        'astigmatism': 'astigmatism',
        'c2_aperture_diameter': 'c2_aperture_diameter',
        'calibrated_defocus_max': 'calibrated_defocus_max',
        'calibrated_defocus_min': 'calibrated_defocus_min',
        'calibrated_magnification': 'calibrated_magnification',
        'cryogen': 'cryogen',
        '_date': 'date',
        'details': 'details',
        'detector_distance': 'detector_distance',
        'electron_beam_tilt_params': 'electron_beam_tilt_params',
        'electron_source': 'electron_source',
        'id': 'id',
        'illumination_mode': 'illumination_mode',
        'microscope_model': 'microscope_model',
        'mode': 'mode',
        'nominal_cs': 'nominal_cs',
        'nominal_defocus_max': 'nominal_defocus_max',
        'nominal_defocus_min': 'nominal_defocus_min',
        'nominal_magnification': 'nominal_magnification',
        'recording_temperature_maximum': 'recording_temperature_maximum',
        'recording_temperature_minimum': 'recording_temperature_minimum',
        'residual_tilt': 'residual_tilt',
        'specimen_holder_model': 'specimen_holder_model',
        'specimen_holder_type': 'specimen_holder_type',
        'specimen_id': 'specimen_id',
        'temperature': 'temperature',
        'tilt_angle_max': 'tilt_angle_max',
        'tilt_angle_min': 'tilt_angle_min'
    }

    def __init__(self, accelerating_voltage=None, alignment_procedure=None, astigmatism=None, c2_aperture_diameter=None, calibrated_defocus_max=None, calibrated_defocus_min=None, calibrated_magnification=None, cryogen=None, _date=None, details=None, detector_distance=None, electron_beam_tilt_params=None, electron_source=None, id=None, illumination_mode=None, microscope_model=None, mode=None, nominal_cs=None, nominal_defocus_max=None, nominal_defocus_min=None, nominal_magnification=None, recording_temperature_maximum=None, recording_temperature_minimum=None, residual_tilt=None, specimen_holder_model=None, specimen_holder_type=None, specimen_id=None, temperature=None, tilt_angle_max=None, tilt_angle_min=None):  # noqa: E501
        """EmImaging - a model defined in Swagger"""  # noqa: E501
        self._accelerating_voltage = None
        self._alignment_procedure = None
        self._astigmatism = None
        self._c2_aperture_diameter = None
        self._calibrated_defocus_max = None
        self._calibrated_defocus_min = None
        self._calibrated_magnification = None
        self._cryogen = None
        self.__date = None
        self._details = None
        self._detector_distance = None
        self._electron_beam_tilt_params = None
        self._electron_source = None
        self._id = None
        self._illumination_mode = None
        self._microscope_model = None
        self._mode = None
        self._nominal_cs = None
        self._nominal_defocus_max = None
        self._nominal_defocus_min = None
        self._nominal_magnification = None
        self._recording_temperature_maximum = None
        self._recording_temperature_minimum = None
        self._residual_tilt = None
        self._specimen_holder_model = None
        self._specimen_holder_type = None
        self._specimen_id = None
        self._temperature = None
        self._tilt_angle_max = None
        self._tilt_angle_min = None
        self.discriminator = None
        if accelerating_voltage is not None:
            self.accelerating_voltage = accelerating_voltage
        if alignment_procedure is not None:
            self.alignment_procedure = alignment_procedure
        if astigmatism is not None:
            self.astigmatism = astigmatism
        if c2_aperture_diameter is not None:
            self.c2_aperture_diameter = c2_aperture_diameter
        if calibrated_defocus_max is not None:
            self.calibrated_defocus_max = calibrated_defocus_max
        if calibrated_defocus_min is not None:
            self.calibrated_defocus_min = calibrated_defocus_min
        if calibrated_magnification is not None:
            self.calibrated_magnification = calibrated_magnification
        if cryogen is not None:
            self.cryogen = cryogen
        if _date is not None:
            self._date = _date
        if details is not None:
            self.details = details
        if detector_distance is not None:
            self.detector_distance = detector_distance
        if electron_beam_tilt_params is not None:
            self.electron_beam_tilt_params = electron_beam_tilt_params
        if electron_source is not None:
            self.electron_source = electron_source
        self.id = id
        if illumination_mode is not None:
            self.illumination_mode = illumination_mode
        if microscope_model is not None:
            self.microscope_model = microscope_model
        if mode is not None:
            self.mode = mode
        if nominal_cs is not None:
            self.nominal_cs = nominal_cs
        if nominal_defocus_max is not None:
            self.nominal_defocus_max = nominal_defocus_max
        if nominal_defocus_min is not None:
            self.nominal_defocus_min = nominal_defocus_min
        if nominal_magnification is not None:
            self.nominal_magnification = nominal_magnification
        if recording_temperature_maximum is not None:
            self.recording_temperature_maximum = recording_temperature_maximum
        if recording_temperature_minimum is not None:
            self.recording_temperature_minimum = recording_temperature_minimum
        if residual_tilt is not None:
            self.residual_tilt = residual_tilt
        if specimen_holder_model is not None:
            self.specimen_holder_model = specimen_holder_model
        if specimen_holder_type is not None:
            self.specimen_holder_type = specimen_holder_type
        if specimen_id is not None:
            self.specimen_id = specimen_id
        if temperature is not None:
            self.temperature = temperature
        if tilt_angle_max is not None:
            self.tilt_angle_max = tilt_angle_max
        if tilt_angle_min is not None:
            self.tilt_angle_min = tilt_angle_min

    @property
    def accelerating_voltage(self):
        """Gets the accelerating_voltage of this EmImaging.  # noqa: E501

        A value of accelerating voltage (in kV) used for imaging.  # noqa: E501

        :return: The accelerating_voltage of this EmImaging.  # noqa: E501
        :rtype: int
        """
        return self._accelerating_voltage

    @accelerating_voltage.setter
    def accelerating_voltage(self, accelerating_voltage):
        """Sets the accelerating_voltage of this EmImaging.

        A value of accelerating voltage (in kV) used for imaging.  # noqa: E501

        :param accelerating_voltage: The accelerating_voltage of this EmImaging.  # noqa: E501
        :type: int
        """

        self._accelerating_voltage = accelerating_voltage

    @property
    def alignment_procedure(self):
        """Gets the alignment_procedure of this EmImaging.  # noqa: E501

        The type of procedure used to align the microscope electron beam.  # noqa: E501

        :return: The alignment_procedure of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._alignment_procedure

    @alignment_procedure.setter
    def alignment_procedure(self, alignment_procedure):
        """Sets the alignment_procedure of this EmImaging.

        The type of procedure used to align the microscope electron beam.  # noqa: E501

        :param alignment_procedure: The alignment_procedure of this EmImaging.  # noqa: E501
        :type: str
        """
        allowed_values = ["BASIC", "COMA FREE", "NONE", "OTHER", "ZEMLIN TABLEAU"]  # noqa: E501
        if alignment_procedure not in allowed_values:
            raise ValueError(
                "Invalid value for `alignment_procedure` ({0}), must be one of {1}"  # noqa: E501
                .format(alignment_procedure, allowed_values)
            )

        self._alignment_procedure = alignment_procedure

    @property
    def astigmatism(self):
        """Gets the astigmatism of this EmImaging.  # noqa: E501

        astigmatism  # noqa: E501

        :return: The astigmatism of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._astigmatism

    @astigmatism.setter
    def astigmatism(self, astigmatism):
        """Sets the astigmatism of this EmImaging.

        astigmatism  # noqa: E501

        :param astigmatism: The astigmatism of this EmImaging.  # noqa: E501
        :type: str
        """

        self._astigmatism = astigmatism

    @property
    def c2_aperture_diameter(self):
        """Gets the c2_aperture_diameter of this EmImaging.  # noqa: E501

        The open diameter of the c2 condenser lens,  in microns.  # noqa: E501

        :return: The c2_aperture_diameter of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._c2_aperture_diameter

    @c2_aperture_diameter.setter
    def c2_aperture_diameter(self, c2_aperture_diameter):
        """Sets the c2_aperture_diameter of this EmImaging.

        The open diameter of the c2 condenser lens,  in microns.  # noqa: E501

        :param c2_aperture_diameter: The c2_aperture_diameter of this EmImaging.  # noqa: E501
        :type: float
        """

        self._c2_aperture_diameter = c2_aperture_diameter

    @property
    def calibrated_defocus_max(self):
        """Gets the calibrated_defocus_max of this EmImaging.  # noqa: E501

        The maximum calibrated defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus.  # noqa: E501

        :return: The calibrated_defocus_max of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._calibrated_defocus_max

    @calibrated_defocus_max.setter
    def calibrated_defocus_max(self, calibrated_defocus_max):
        """Sets the calibrated_defocus_max of this EmImaging.

        The maximum calibrated defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus.  # noqa: E501

        :param calibrated_defocus_max: The calibrated_defocus_max of this EmImaging.  # noqa: E501
        :type: float
        """

        self._calibrated_defocus_max = calibrated_defocus_max

    @property
    def calibrated_defocus_min(self):
        """Gets the calibrated_defocus_min of this EmImaging.  # noqa: E501

        The minimum calibrated defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus.  # noqa: E501

        :return: The calibrated_defocus_min of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._calibrated_defocus_min

    @calibrated_defocus_min.setter
    def calibrated_defocus_min(self, calibrated_defocus_min):
        """Sets the calibrated_defocus_min of this EmImaging.

        The minimum calibrated defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus.  # noqa: E501

        :param calibrated_defocus_min: The calibrated_defocus_min of this EmImaging.  # noqa: E501
        :type: float
        """

        self._calibrated_defocus_min = calibrated_defocus_min

    @property
    def calibrated_magnification(self):
        """Gets the calibrated_magnification of this EmImaging.  # noqa: E501

        The magnification value obtained for a known standard just  prior to, during or just after the imaging experiment.  # noqa: E501

        :return: The calibrated_magnification of this EmImaging.  # noqa: E501
        :rtype: int
        """
        return self._calibrated_magnification

    @calibrated_magnification.setter
    def calibrated_magnification(self, calibrated_magnification):
        """Sets the calibrated_magnification of this EmImaging.

        The magnification value obtained for a known standard just  prior to, during or just after the imaging experiment.  # noqa: E501

        :param calibrated_magnification: The calibrated_magnification of this EmImaging.  # noqa: E501
        :type: int
        """

        self._calibrated_magnification = calibrated_magnification

    @property
    def cryogen(self):
        """Gets the cryogen of this EmImaging.  # noqa: E501

        Cryogen type used to maintain the specimen stage temperature during imaging  in the microscope.  # noqa: E501

        :return: The cryogen of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._cryogen

    @cryogen.setter
    def cryogen(self, cryogen):
        """Sets the cryogen of this EmImaging.

        Cryogen type used to maintain the specimen stage temperature during imaging  in the microscope.  # noqa: E501

        :param cryogen: The cryogen of this EmImaging.  # noqa: E501
        :type: str
        """
        allowed_values = ["HELIUM", "NITROGEN"]  # noqa: E501
        if cryogen not in allowed_values:
            raise ValueError(
                "Invalid value for `cryogen` ({0}), must be one of {1}"  # noqa: E501
                .format(cryogen, allowed_values)
            )

        self._cryogen = cryogen

    @property
    def _date(self):
        """Gets the _date of this EmImaging.  # noqa: E501

        Date (YYYY-MM-DD) of imaging experiment or the date at which  a series of experiments began.  # noqa: E501

        :return: The _date of this EmImaging.  # noqa: E501
        :rtype: datetime
        """
        return self.__date

    @_date.setter
    def _date(self, _date):
        """Sets the _date of this EmImaging.

        Date (YYYY-MM-DD) of imaging experiment or the date at which  a series of experiments began.  # noqa: E501

        :param _date: The _date of this EmImaging.  # noqa: E501
        :type: datetime
        """

        self.__date = _date

    @property
    def details(self):
        """Gets the details of this EmImaging.  # noqa: E501

        Any additional imaging details.  # noqa: E501

        :return: The details of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._details

    @details.setter
    def details(self, details):
        """Sets the details of this EmImaging.

        Any additional imaging details.  # noqa: E501

        :param details: The details of this EmImaging.  # noqa: E501
        :type: str
        """

        self._details = details

    @property
    def detector_distance(self):
        """Gets the detector_distance of this EmImaging.  # noqa: E501

        The camera length (in millimeters). The camera length is the  product of the objective focal length and the combined magnification  of the intermediate and projector lenses when the microscope is  operated in the diffraction mode.  # noqa: E501

        :return: The detector_distance of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._detector_distance

    @detector_distance.setter
    def detector_distance(self, detector_distance):
        """Sets the detector_distance of this EmImaging.

        The camera length (in millimeters). The camera length is the  product of the objective focal length and the combined magnification  of the intermediate and projector lenses when the microscope is  operated in the diffraction mode.  # noqa: E501

        :param detector_distance: The detector_distance of this EmImaging.  # noqa: E501
        :type: float
        """

        self._detector_distance = detector_distance

    @property
    def electron_beam_tilt_params(self):
        """Gets the electron_beam_tilt_params of this EmImaging.  # noqa: E501

        electron beam tilt params  # noqa: E501

        :return: The electron_beam_tilt_params of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._electron_beam_tilt_params

    @electron_beam_tilt_params.setter
    def electron_beam_tilt_params(self, electron_beam_tilt_params):
        """Sets the electron_beam_tilt_params of this EmImaging.

        electron beam tilt params  # noqa: E501

        :param electron_beam_tilt_params: The electron_beam_tilt_params of this EmImaging.  # noqa: E501
        :type: str
        """

        self._electron_beam_tilt_params = electron_beam_tilt_params

    @property
    def electron_source(self):
        """Gets the electron_source of this EmImaging.  # noqa: E501

        The source of electrons. The electron gun.  # noqa: E501

        :return: The electron_source of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._electron_source

    @electron_source.setter
    def electron_source(self, electron_source):
        """Sets the electron_source of this EmImaging.

        The source of electrons. The electron gun.  # noqa: E501

        :param electron_source: The electron_source of this EmImaging.  # noqa: E501
        :type: str
        """

        self._electron_source = electron_source

    @property
    def id(self):
        """Gets the id of this EmImaging.  # noqa: E501

        PRIMARY KEY  # noqa: E501

        :return: The id of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this EmImaging.

        PRIMARY KEY  # noqa: E501

        :param id: The id of this EmImaging.  # noqa: E501
        :type: str
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def illumination_mode(self):
        """Gets the illumination_mode of this EmImaging.  # noqa: E501

        The mode of illumination.  # noqa: E501

        :return: The illumination_mode of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._illumination_mode

    @illumination_mode.setter
    def illumination_mode(self, illumination_mode):
        """Sets the illumination_mode of this EmImaging.

        The mode of illumination.  # noqa: E501

        :param illumination_mode: The illumination_mode of this EmImaging.  # noqa: E501
        :type: str
        """
        allowed_values = ["FLOOD BEAM", "OTHER", "SPOT SCAN"]  # noqa: E501
        if illumination_mode not in allowed_values:
            raise ValueError(
                "Invalid value for `illumination_mode` ({0}), must be one of {1}"  # noqa: E501
                .format(illumination_mode, allowed_values)
            )

        self._illumination_mode = illumination_mode

    @property
    def microscope_model(self):
        """Gets the microscope_model of this EmImaging.  # noqa: E501

        The name of the model of microscope.  # noqa: E501

        :return: The microscope_model of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._microscope_model

    @microscope_model.setter
    def microscope_model(self, microscope_model):
        """Sets the microscope_model of this EmImaging.

        The name of the model of microscope.  # noqa: E501

        :param microscope_model: The microscope_model of this EmImaging.  # noqa: E501
        :type: str
        """
        allowed_values = ["FEI MORGAGNI", "FEI POLARA 300", "FEI TALOS ARCTICA", "FEI TECNAI 10", "FEI TECNAI 12", "FEI TECNAI 20", "FEI TECNAI ARCTICA", "FEI TECNAI F20", "FEI TECNAI F30", "FEI TECNAI SPHERA", "FEI TECNAI SPIRIT", "FEI TITAN", "FEI TITAN KRIOS", "FEI/PHILIPS CM10", "FEI/PHILIPS CM12", "FEI/PHILIPS CM120T", "FEI/PHILIPS CM200FEG", "FEI/PHILIPS CM200FEG/SOPHIE", "FEI/PHILIPS CM200FEG/ST", "FEI/PHILIPS CM200FEG/UT", "FEI/PHILIPS CM200T", "FEI/PHILIPS CM300FEG/HE", "FEI/PHILIPS CM300FEG/ST", "FEI/PHILIPS CM300FEG/T", "FEI/PHILIPS EM400", "FEI/PHILIPS EM420", "HITACHI EF2000", "HITACHI EF3000", "HITACHI H-9500SD", "HITACHI H3000 UHVEM", "HITACHI H7600", "HITACHI HF2000", "HITACHI HF3000", "JEOL 1000EES", "JEOL 100B", "JEOL 100CX", "JEOL 1010", "JEOL 1200", "JEOL 1200EX", "JEOL 1200EXII", "JEOL 1230", "JEOL 1400", "JEOL 2000EX", "JEOL 2000EXII", "JEOL 2010", "JEOL 2010F", "JEOL 2010HC", "JEOL 2010HT", "JEOL 2010UHR", "JEOL 2011", "JEOL 2100", "JEOL 2100F", "JEOL 2200FS", "JEOL 2200FSC", "JEOL 3000SFF", "JEOL 3100FEF", "JEOL 3100FFC", "JEOL 3200FS", "JEOL 3200FSC", "JEOL 4000", "JEOL 4000EX", "JEOL CRYO ARM 200", "JEOL CRYO ARM 300", "JEOL KYOTO-3000SFF", "SIEMENS SULEIKA", "TFS GLACIOS", "TFS KRIOS", "TFS TALOS", "TFS TALOS F200C", "TFS TALOS L120C", "TFS TUNDRA", "ZEISS LEO912", "ZEISS LIBRA120PLUS"]  # noqa: E501
        if microscope_model not in allowed_values:
            raise ValueError(
                "Invalid value for `microscope_model` ({0}), must be one of {1}"  # noqa: E501
                .format(microscope_model, allowed_values)
            )

        self._microscope_model = microscope_model

    @property
    def mode(self):
        """Gets the mode of this EmImaging.  # noqa: E501

        The mode of imaging.  # noqa: E501

        :return: The mode of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        """Sets the mode of this EmImaging.

        The mode of imaging.  # noqa: E501

        :param mode: The mode of this EmImaging.  # noqa: E501
        :type: str
        """
        allowed_values = ["BRIGHT FIELD", "DARK FIELD", "DIFFRACTION", "OTHER"]  # noqa: E501
        if mode not in allowed_values:
            raise ValueError(
                "Invalid value for `mode` ({0}), must be one of {1}"  # noqa: E501
                .format(mode, allowed_values)
            )

        self._mode = mode

    @property
    def nominal_cs(self):
        """Gets the nominal_cs of this EmImaging.  # noqa: E501

        The spherical aberration coefficient (Cs) in millimeters,  of the objective lens.  # noqa: E501

        :return: The nominal_cs of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._nominal_cs

    @nominal_cs.setter
    def nominal_cs(self, nominal_cs):
        """Sets the nominal_cs of this EmImaging.

        The spherical aberration coefficient (Cs) in millimeters,  of the objective lens.  # noqa: E501

        :param nominal_cs: The nominal_cs of this EmImaging.  # noqa: E501
        :type: float
        """

        self._nominal_cs = nominal_cs

    @property
    def nominal_defocus_max(self):
        """Gets the nominal_defocus_max of this EmImaging.  # noqa: E501

        The maximum defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus.  # noqa: E501

        :return: The nominal_defocus_max of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._nominal_defocus_max

    @nominal_defocus_max.setter
    def nominal_defocus_max(self, nominal_defocus_max):
        """Sets the nominal_defocus_max of this EmImaging.

        The maximum defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus.  # noqa: E501

        :param nominal_defocus_max: The nominal_defocus_max of this EmImaging.  # noqa: E501
        :type: float
        """

        self._nominal_defocus_max = nominal_defocus_max

    @property
    def nominal_defocus_min(self):
        """Gets the nominal_defocus_min of this EmImaging.  # noqa: E501

        The minimum defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus.  # noqa: E501

        :return: The nominal_defocus_min of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._nominal_defocus_min

    @nominal_defocus_min.setter
    def nominal_defocus_min(self, nominal_defocus_min):
        """Sets the nominal_defocus_min of this EmImaging.

        The minimum defocus value of the objective lens (in nanometers) used  to obtain the recorded images. Negative values refer to overfocus.  # noqa: E501

        :param nominal_defocus_min: The nominal_defocus_min of this EmImaging.  # noqa: E501
        :type: float
        """

        self._nominal_defocus_min = nominal_defocus_min

    @property
    def nominal_magnification(self):
        """Gets the nominal_magnification of this EmImaging.  # noqa: E501

        The magnification indicated by the microscope readout.  # noqa: E501

        :return: The nominal_magnification of this EmImaging.  # noqa: E501
        :rtype: int
        """
        return self._nominal_magnification

    @nominal_magnification.setter
    def nominal_magnification(self, nominal_magnification):
        """Sets the nominal_magnification of this EmImaging.

        The magnification indicated by the microscope readout.  # noqa: E501

        :param nominal_magnification: The nominal_magnification of this EmImaging.  # noqa: E501
        :type: int
        """

        self._nominal_magnification = nominal_magnification

    @property
    def recording_temperature_maximum(self):
        """Gets the recording_temperature_maximum of this EmImaging.  # noqa: E501

        The specimen temperature maximum (kelvin) for the duration  of imaging.  # noqa: E501

        :return: The recording_temperature_maximum of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._recording_temperature_maximum

    @recording_temperature_maximum.setter
    def recording_temperature_maximum(self, recording_temperature_maximum):
        """Sets the recording_temperature_maximum of this EmImaging.

        The specimen temperature maximum (kelvin) for the duration  of imaging.  # noqa: E501

        :param recording_temperature_maximum: The recording_temperature_maximum of this EmImaging.  # noqa: E501
        :type: float
        """

        self._recording_temperature_maximum = recording_temperature_maximum

    @property
    def recording_temperature_minimum(self):
        """Gets the recording_temperature_minimum of this EmImaging.  # noqa: E501

        The specimen temperature minimum (kelvin) for the duration  of imaging.  # noqa: E501

        :return: The recording_temperature_minimum of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._recording_temperature_minimum

    @recording_temperature_minimum.setter
    def recording_temperature_minimum(self, recording_temperature_minimum):
        """Sets the recording_temperature_minimum of this EmImaging.

        The specimen temperature minimum (kelvin) for the duration  of imaging.  # noqa: E501

        :param recording_temperature_minimum: The recording_temperature_minimum of this EmImaging.  # noqa: E501
        :type: float
        """

        self._recording_temperature_minimum = recording_temperature_minimum

    @property
    def residual_tilt(self):
        """Gets the residual_tilt of this EmImaging.  # noqa: E501

        Residual tilt of the electron beam (in miliradians)  # noqa: E501

        :return: The residual_tilt of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._residual_tilt

    @residual_tilt.setter
    def residual_tilt(self, residual_tilt):
        """Sets the residual_tilt of this EmImaging.

        Residual tilt of the electron beam (in miliradians)  # noqa: E501

        :param residual_tilt: The residual_tilt of this EmImaging.  # noqa: E501
        :type: float
        """

        self._residual_tilt = residual_tilt

    @property
    def specimen_holder_model(self):
        """Gets the specimen_holder_model of this EmImaging.  # noqa: E501

        The name of the model of specimen holder used during imaging.  # noqa: E501

        :return: The specimen_holder_model of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._specimen_holder_model

    @specimen_holder_model.setter
    def specimen_holder_model(self, specimen_holder_model):
        """Sets the specimen_holder_model of this EmImaging.

        The name of the model of specimen holder used during imaging.  # noqa: E501

        :param specimen_holder_model: The specimen_holder_model of this EmImaging.  # noqa: E501
        :type: str
        """
        allowed_values = ["FEI TITAN KRIOS AUTOGRID HOLDER", "FISCHIONE 2550", "FISCHIONE INSTRUMENTS DUAL AXIS TOMOGRAPHY HOLDER", "GATAN 626 SINGLE TILT LIQUID NITROGEN CRYO TRANSFER HOLDER", "GATAN 910 MULTI-SPECIMEN SINGLE TILT CRYO TRANSFER HOLDER", "GATAN 914 HIGH TILT LIQUID NITROGEN CRYO TRANSFER TOMOGRAPHY HOLDER", "GATAN 915 DOUBLE TILT LIQUID NITROGEN CRYO TRANSFER HOLDER", "GATAN CHDT 3504 DOUBLE TILT HIGH RESOLUTION NITROGEN COOLING HOLDER", "GATAN CT3500 SINGLE TILT LIQUID NITROGEN CRYO TRANSFER HOLDER", "GATAN CT3500TR SINGLE TILT ROTATION LIQUID NITROGEN CRYO TRANSFER HOLDER", "GATAN ELSA 698 SINGLE TILT LIQUID NITROGEN CRYO TRANSFER HOLDER", "GATAN HC 3500 SINGLE TILT HEATING/NITROGEN COOLING HOLDER", "GATAN HCHDT 3010 DOUBLE TILT HIGH RESOLUTION HELIUM COOLING HOLDER", "GATAN HCHST 3008 SINGLE TILT HIGH RESOLUTION HELIUM COOLING HOLDER", "GATAN HELIUM", "GATAN LIQUID NITROGEN", "GATAN UHRST 3500 SINGLE TILT ULTRA HIGH RESOLUTION NITROGEN COOLING HOLDER", "GATAN ULTDT ULTRA LOW TEMPERATURE DOUBLE TILT HELIUM COOLING HOLDER", "GATAN ULTST ULTRA LOW TEMPERATURE SINGLE TILT HELIUM COOLING HOLDER", "HOME BUILD", "JEOL", "JEOL 3200FSC CRYOHOLDER", "JEOL CRYOSPECPORTER", "OTHER", "PHILIPS ROTATION HOLDER", "SIDE ENTRY, EUCENTRIC"]  # noqa: E501
        if specimen_holder_model not in allowed_values:
            raise ValueError(
                "Invalid value for `specimen_holder_model` ({0}), must be one of {1}"  # noqa: E501
                .format(specimen_holder_model, allowed_values)
            )

        self._specimen_holder_model = specimen_holder_model

    @property
    def specimen_holder_type(self):
        """Gets the specimen_holder_type of this EmImaging.  # noqa: E501

        The type of specimen holder used during imaging.  # noqa: E501

        :return: The specimen_holder_type of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._specimen_holder_type

    @specimen_holder_type.setter
    def specimen_holder_type(self, specimen_holder_type):
        """Sets the specimen_holder_type of this EmImaging.

        The type of specimen holder used during imaging.  # noqa: E501

        :param specimen_holder_type: The specimen_holder_type of this EmImaging.  # noqa: E501
        :type: str
        """

        self._specimen_holder_type = specimen_holder_type

    @property
    def specimen_id(self):
        """Gets the specimen_id of this EmImaging.  # noqa: E501

        Foreign key to the EM_SPECIMEN category  # noqa: E501

        :return: The specimen_id of this EmImaging.  # noqa: E501
        :rtype: str
        """
        return self._specimen_id

    @specimen_id.setter
    def specimen_id(self, specimen_id):
        """Sets the specimen_id of this EmImaging.

        Foreign key to the EM_SPECIMEN category  # noqa: E501

        :param specimen_id: The specimen_id of this EmImaging.  # noqa: E501
        :type: str
        """

        self._specimen_id = specimen_id

    @property
    def temperature(self):
        """Gets the temperature of this EmImaging.  # noqa: E501

        The mean specimen stage temperature (in kelvin) during imaging  in the microscope.  # noqa: E501

        :return: The temperature of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        """Sets the temperature of this EmImaging.

        The mean specimen stage temperature (in kelvin) during imaging  in the microscope.  # noqa: E501

        :param temperature: The temperature of this EmImaging.  # noqa: E501
        :type: float
        """

        self._temperature = temperature

    @property
    def tilt_angle_max(self):
        """Gets the tilt_angle_max of this EmImaging.  # noqa: E501

        The maximum angle at which the specimen was tilted to obtain  recorded images.  # noqa: E501

        :return: The tilt_angle_max of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._tilt_angle_max

    @tilt_angle_max.setter
    def tilt_angle_max(self, tilt_angle_max):
        """Sets the tilt_angle_max of this EmImaging.

        The maximum angle at which the specimen was tilted to obtain  recorded images.  # noqa: E501

        :param tilt_angle_max: The tilt_angle_max of this EmImaging.  # noqa: E501
        :type: float
        """

        self._tilt_angle_max = tilt_angle_max

    @property
    def tilt_angle_min(self):
        """Gets the tilt_angle_min of this EmImaging.  # noqa: E501

        The minimum angle at which the specimen was tilted to obtain  recorded images.  # noqa: E501

        :return: The tilt_angle_min of this EmImaging.  # noqa: E501
        :rtype: float
        """
        return self._tilt_angle_min

    @tilt_angle_min.setter
    def tilt_angle_min(self, tilt_angle_min):
        """Sets the tilt_angle_min of this EmImaging.

        The minimum angle at which the specimen was tilted to obtain  recorded images.  # noqa: E501

        :param tilt_angle_min: The tilt_angle_min of this EmImaging.  # noqa: E501
        :type: float
        """

        self._tilt_angle_min = tilt_angle_min

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
        if issubclass(EmImaging, dict):
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
        if not isinstance(other, EmImaging):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
