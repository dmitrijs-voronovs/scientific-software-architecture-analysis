from pydantic import BaseModel, Field
from typing import Literal, Annotated


class PingEchoModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Detect Faults"]
    tactic: Literal["Ping/Echo"]
    response: str


class MonitorModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Detect Faults"]
    tactic: Literal["Monitor"]
    response: str


class HeartbeatModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Detect Faults"]
    tactic: Literal["Heartbeat"]
    response: str


class TimestampModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Detect Faults"]
    tactic: Literal["Timestamp"]
    response: str


class SanityCheckingModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Detect Faults"]
    tactic: Literal["Sanity Checking"]
    response: str


class ConditionMonitoringModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Detect Faults"]
    tactic: Literal["Condition Monitoring"]
    response: str


class VotingModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Detect Faults"]
    tactic: Literal["Voting"]
    response: str


class ExceptionDetectionModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Detect Faults"]
    tactic: Literal["Exception Detection"]
    response: str


class SelfTestModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Detect Faults"]
    tactic: Literal["Self-Test"]
    response: str


DetectFaultsModel = Annotated[PingEchoModel | MonitorModel | HeartbeatModel | TimestampModel | SanityCheckingModel | ConditionMonitoringModel | VotingModel | ExceptionDetectionModel | SelfTestModel, Field(discriminator='tactic')]

class ActiveRedundancyModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Preparation and Repair"]
    tactic: Literal["Active Redundancy"]
    response: str


class PassiveRedundancyModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Preparation and Repair"]
    tactic: Literal["Passive Redundancy"]
    response: str


class SpareModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Preparation and Repair"]
    tactic: Literal["Spare"]
    response: str


class ExceptionHandlingModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Preparation and Repair"]
    tactic: Literal["Exception Handling"]
    response: str


class RollbackModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Preparation and Repair"]
    tactic: Literal["Rollback"]
    response: str


class SoftwareUpgradeModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Preparation and Repair"]
    tactic: Literal["Software Upgrade"]
    response: str


class RetryModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Preparation and Repair"]
    tactic: Literal["Retry"]
    response: str


class IgnoreFaultyBehaviorModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Preparation and Repair"]
    tactic: Literal["Ignore Faulty Behavior"]
    response: str


class DegradationModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Preparation and Repair"]
    tactic: Literal["Degradation"]
    response: str


class ReconfigurationModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Preparation and Repair"]
    tactic: Literal["Reconfiguration"]
    response: str


RecoverFromFaults_preparationAndRepairModel = Annotated[ActiveRedundancyModel | PassiveRedundancyModel | SpareModel | ExceptionHandlingModel | RollbackModel | SoftwareUpgradeModel | RetryModel | IgnoreFaultyBehaviorModel | DegradationModel | ReconfigurationModel, Field(discriminator='tactic')]

class ShadowModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Reintroduction"]
    tactic: Literal["Shadow"]
    response: str


class StateResynchronizationModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Reintroduction"]
    tactic: Literal["State Resynchronization"]
    response: str


class EscalatingRestartModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Reintroduction"]
    tactic: Literal["Escalating Restart"]
    response: str


class NonStopForwardingModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Recover from Faults_Reintroduction"]
    tactic: Literal["Non-Stop Forwarding"]
    response: str


RecoverFromFaults_reintroductionModel = Annotated[ShadowModel | StateResynchronizationModel | EscalatingRestartModel | NonStopForwardingModel, Field(discriminator='tactic')]

class RemovalFromServiceModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Prevent Faults"]
    tactic: Literal["Removal from Service"]
    response: str


class TransactionsModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Prevent Faults"]
    tactic: Literal["Transactions"]
    response: str


class PredictiveModelModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Prevent Faults"]
    tactic: Literal["Predictive Model"]
    response: str


class ExceptionPreventionModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Prevent Faults"]
    tactic: Literal["Exception Prevention"]
    response: str


class IncreaseCompetenceSetModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Prevent Faults"]
    tactic: Literal["Increase Competence Set"]
    response: str


PreventFaultsModel = Annotated[RemovalFromServiceModel | TransactionsModel | PredictiveModelModel | ExceptionPreventionModel | IncreaseCompetenceSetModel, Field(discriminator='tactic')]

AvailabilityModel = Annotated[DetectFaultsModel | RecoverFromFaults_preparationAndRepairModel | RecoverFromFaults_reintroductionModel | PreventFaultsModel, Field(discriminator='tactic_category')]

class DiscoverServiceModel(BaseModel):
    quality_attribute: Literal["Interoperability"]
    tactic_category: Literal["Locate"]
    tactic: Literal["Discover Service"]
    response: str


LocateModel = Annotated[DiscoverServiceModel, Field(discriminator='tactic')]

class OrchestrateModel(BaseModel):
    quality_attribute: Literal["Interoperability"]
    tactic_category: Literal["Manage Interfaces"]
    tactic: Literal["Orchestrate"]
    response: str


class TailorInterfaceModel(BaseModel):
    quality_attribute: Literal["Interoperability"]
    tactic_category: Literal["Manage Interfaces"]
    tactic: Literal["Tailor Interface"]
    response: str


ManageInterfacesModel = Annotated[OrchestrateModel | TailorInterfaceModel, Field(discriminator='tactic')]

InteroperabilityModel = Annotated[LocateModel | ManageInterfacesModel, Field(discriminator='tactic_category')]

class SplitModuleModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Reduce Size of a Module"]
    tactic: Literal["Split Module"]
    response: str


ReduceSizeOfAModuleModel = Annotated[SplitModuleModel, Field(discriminator='tactic')]

class IncreaseSemanticCoherenceModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Increase Cohesion"]
    tactic: Literal["Increase Semantic Coherence"]
    response: str


IncreaseCohesionModel = Annotated[IncreaseSemanticCoherenceModel, Field(discriminator='tactic')]

class EncapsulateModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Reduce Coupling"]
    tactic: Literal["Encapsulate"]
    response: str


class UseAnIntermediaryModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Reduce Coupling"]
    tactic: Literal["Use an Intermediary"]
    response: str


class RestrictDependenciesModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Reduce Coupling"]
    tactic: Literal["Restrict Dependencies"]
    response: str


class RefactorModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Reduce Coupling"]
    tactic: Literal["Refactor"]
    response: str


class AbstractCommonServicesModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Reduce Coupling"]
    tactic: Literal["Abstract Common Services"]
    response: str


ReduceCouplingModel = Annotated[EncapsulateModel | UseAnIntermediaryModel | RestrictDependenciesModel | RefactorModel | AbstractCommonServicesModel, Field(discriminator='tactic')]

class ComponentReplacementModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Component Replacement"]
    response: str


class CompileTimeParameterizationModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Compile-time Parameterization"]
    response: str


class AspectsModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Aspects"]
    response: str


class ConfigurationTimeBindingModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Configuration-time Binding"]
    response: str


class ResourceFilesModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Resource Files"]
    response: str


class RuntimeRegistrationModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Runtime Registration"]
    response: str


class DynamicLookupModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Dynamic Lookup"]
    response: str


class InterpretParametersModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Interpret Parameters"]
    response: str


class StartupTimeBindingModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Startup Time Binding"]
    response: str


class NameServersModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Name Servers"]
    response: str


class PlugInsModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Plug-ins"]
    response: str


class PublishSubscribeModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Publish-Subscribe"]
    response: str


class SharedRepositoriesModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Shared Repositories"]
    response: str


class PolymorphismModel(BaseModel):
    quality_attribute: Literal["Modifiability"]
    tactic_category: Literal["Defer Binding"]
    tactic: Literal["Polymorphism"]
    response: str


DeferBindingModel = Annotated[ComponentReplacementModel | CompileTimeParameterizationModel | AspectsModel | ConfigurationTimeBindingModel | ResourceFilesModel | RuntimeRegistrationModel | DynamicLookupModel | InterpretParametersModel | StartupTimeBindingModel | NameServersModel | PlugInsModel | PublishSubscribeModel | SharedRepositoriesModel | PolymorphismModel, Field(discriminator='tactic')]

ModifiabilityModel = Annotated[ReduceSizeOfAModuleModel | IncreaseCohesionModel | ReduceCouplingModel | DeferBindingModel, Field(discriminator='tactic_category')]

class ManageSamplingRateModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Control Resource Demand"]
    tactic: Literal["Manage Sampling Rate"]
    response: str


class LimitEventResponseModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Control Resource Demand"]
    tactic: Literal["Limit Event Response"]
    response: str


class PrioritizeEventsModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Control Resource Demand"]
    tactic: Literal["Prioritize Events"]
    response: str


class ReduceOverheadModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Control Resource Demand"]
    tactic: Literal["Reduce Overhead"]
    response: str


class BoundExecutionTimesModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Control Resource Demand"]
    tactic: Literal["Bound Execution Times"]
    response: str


class IncreaseResourceEfficiencyModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Control Resource Demand"]
    tactic: Literal["Increase Resource Efficiency"]
    response: str


ControlResourceDemandModel = Annotated[ManageSamplingRateModel | LimitEventResponseModel | PrioritizeEventsModel | ReduceOverheadModel | BoundExecutionTimesModel | IncreaseResourceEfficiencyModel, Field(discriminator='tactic')]

class IncreaseResourcesModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Manage Resources"]
    tactic: Literal["Increase Resources"]
    response: str


class IntroduceConcurrencyModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Manage Resources"]
    tactic: Literal["Introduce Concurrency"]
    response: str


class MaintainMultipleCopiesOfComputationsModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Manage Resources"]
    tactic: Literal["Maintain Multiple Copies of Computations"]
    response: str


class MaintainMultipleCopiesOfDataModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Manage Resources"]
    tactic: Literal["Maintain Multiple Copies of Data"]
    response: str


class BoundQueueSizesModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Manage Resources"]
    tactic: Literal["Bound Queue Sizes"]
    response: str


class ScheduleResourcesModel(BaseModel):
    quality_attribute: Literal["Performance"]
    tactic_category: Literal["Manage Resources"]
    tactic: Literal["Schedule Resources"]
    response: str


ManageResourcesModel = Annotated[IncreaseResourcesModel | IntroduceConcurrencyModel | MaintainMultipleCopiesOfComputationsModel | MaintainMultipleCopiesOfDataModel | BoundQueueSizesModel | ScheduleResourcesModel, Field(discriminator='tactic')]

PerformanceModel = Annotated[ControlResourceDemandModel | ManageResourcesModel, Field(discriminator='tactic_category')]

class DetectIntrusionModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Detect Attacks"]
    tactic: Literal["Detect Intrusion"]
    response: str


class DetectServiceDenialModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Detect Attacks"]
    tactic: Literal["Detect Service Denial"]
    response: str


class VerifyMessageIntegrityModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Detect Attacks"]
    tactic: Literal["Verify Message Integrity"]
    response: str


class DetectMessageDelayModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Detect Attacks"]
    tactic: Literal["Detect Message Delay"]
    response: str


DetectAttacksModel = Annotated[DetectIntrusionModel | DetectServiceDenialModel | VerifyMessageIntegrityModel | DetectMessageDelayModel, Field(discriminator='tactic')]

class IdentifyActorsModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Resist Attacks"]
    tactic: Literal["Identify Actors"]
    response: str


class AuthenticateActorsModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Resist Attacks"]
    tactic: Literal["Authenticate Actors"]
    response: str


class AuthorizeActorsModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Resist Attacks"]
    tactic: Literal["Authorize Actors"]
    response: str


class LimitAccessModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Resist Attacks"]
    tactic: Literal["Limit Access"]
    response: str


class LimitExposureModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Resist Attacks"]
    tactic: Literal["Limit Exposure"]
    response: str


class EncryptDataModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Resist Attacks"]
    tactic: Literal["Encrypt Data"]
    response: str


class SeparateEntitiesModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Resist Attacks"]
    tactic: Literal["Separate Entities"]
    response: str


class ChangeDefaultSettingsModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Resist Attacks"]
    tactic: Literal["Change Default Settings"]
    response: str


ResistAttacksModel = Annotated[IdentifyActorsModel | AuthenticateActorsModel | AuthorizeActorsModel | LimitAccessModel | LimitExposureModel | EncryptDataModel | SeparateEntitiesModel | ChangeDefaultSettingsModel, Field(discriminator='tactic')]

class RevokeAccessModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["React to Attacks"]
    tactic: Literal["Revoke Access"]
    response: str


class LockComputerModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["React to Attacks"]
    tactic: Literal["Lock Computer"]
    response: str


class InformActorsModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["React to Attacks"]
    tactic: Literal["Inform Actors"]
    response: str


ReactToAttacksModel = Annotated[RevokeAccessModel | LockComputerModel | InformActorsModel, Field(discriminator='tactic')]

class MaintainAuditTrailModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Recover from Attacks"]
    tactic: Literal["Maintain Audit Trail"]
    response: str


class RestoreModel(BaseModel):
    quality_attribute: Literal["Security"]
    tactic_category: Literal["Recover from Attacks"]
    tactic: Literal["Restore"]
    response: str


RecoverFromAttacksModel = Annotated[MaintainAuditTrailModel | RestoreModel, Field(discriminator='tactic')]

SecurityModel = Annotated[DetectAttacksModel | ResistAttacksModel | ReactToAttacksModel | RecoverFromAttacksModel, Field(discriminator='tactic_category')]

class SpecializedInterfacesModel(BaseModel):
    quality_attribute: Literal["Testability"]
    tactic_category: Literal["Control and Observe System State"]
    tactic: Literal["Specialized Interfaces"]
    response: str


class RecordPlaybackModel(BaseModel):
    quality_attribute: Literal["Testability"]
    tactic_category: Literal["Control and Observe System State"]
    tactic: Literal["Record/Playback"]
    response: str


class LocalizeStateStorageModel(BaseModel):
    quality_attribute: Literal["Testability"]
    tactic_category: Literal["Control and Observe System State"]
    tactic: Literal["Localize State Storage"]
    response: str


class AbstractDataSourcesModel(BaseModel):
    quality_attribute: Literal["Testability"]
    tactic_category: Literal["Control and Observe System State"]
    tactic: Literal["Abstract Data Sources"]
    response: str


class SandboxModel(BaseModel):
    quality_attribute: Literal["Testability"]
    tactic_category: Literal["Control and Observe System State"]
    tactic: Literal["Sandbox"]
    response: str


class ExecutableAssertionsModel(BaseModel):
    quality_attribute: Literal["Testability"]
    tactic_category: Literal["Control and Observe System State"]
    tactic: Literal["Executable Assertions"]
    response: str


ControlAndObserveSystemStateModel = Annotated[SpecializedInterfacesModel | RecordPlaybackModel | LocalizeStateStorageModel | AbstractDataSourcesModel | SandboxModel | ExecutableAssertionsModel, Field(discriminator='tactic')]

class LimitStructuralComplexityModel(BaseModel):
    quality_attribute: Literal["Testability"]
    tactic_category: Literal["Limit Complexity"]
    tactic: Literal["Limit Structural Complexity"]
    response: str


class LimitNondeterminismModel(BaseModel):
    quality_attribute: Literal["Testability"]
    tactic_category: Literal["Limit Complexity"]
    tactic: Literal["Limit Nondeterminism"]
    response: str


LimitComplexityModel = Annotated[LimitStructuralComplexityModel | LimitNondeterminismModel, Field(discriminator='tactic')]

TestabilityModel = Annotated[ControlAndObserveSystemStateModel | LimitComplexityModel, Field(discriminator='tactic_category')]

class IncreaseSemanticCoherenceModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Separate the User Interface"]
    tactic: Literal["Increase semantic coherence"]
    response: str


class EncapsulateModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Separate the User Interface"]
    tactic: Literal["Encapsulate"]
    response: str


class CoLocateRelatedResponsibilitiesModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Separate the User Interface"]
    tactic: Literal["Co-locate related responsibilities"]
    response: str


class RestrictDependenciesModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Separate the User Interface"]
    tactic: Literal["Restrict dependencies"]
    response: str


class DeferBindingModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Separate the User Interface"]
    tactic: Literal["Defer binding"]
    response: str


SeparateTheUserInterfaceModel = Annotated[IncreaseSemanticCoherenceModel | EncapsulateModel | CoLocateRelatedResponsibilitiesModel | RestrictDependenciesModel | DeferBindingModel, Field(discriminator='tactic')]

class CancelModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Support User Initiative"]
    tactic: Literal["Cancel"]
    response: str


class UndoModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Support User Initiative"]
    tactic: Literal["Undo"]
    response: str


class PauseResumeModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Support User Initiative"]
    tactic: Literal["Pause/resume"]
    response: str


class AggregateModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Support User Initiative"]
    tactic: Literal["Aggregate"]
    response: str


SupportUserInitiativeModel = Annotated[CancelModel | UndoModel | PauseResumeModel | AggregateModel, Field(discriminator='tactic')]

class MaintainTaskModelModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Support System Initiative"]
    tactic: Literal["Maintain task model"]
    response: str


class MaintainUserModelModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Support System Initiative"]
    tactic: Literal["Maintain user model"]
    response: str


class MaintainSystemModelModel(BaseModel):
    quality_attribute: Literal["Usability"]
    tactic_category: Literal["Support System Initiative"]
    tactic: Literal["Maintain system model"]
    response: str


SupportSystemInitiativeModel = Annotated[MaintainTaskModelModel | MaintainUserModelModel | MaintainSystemModelModel, Field(discriminator='tactic')]

UsabilityModel = Annotated[SeparateTheUserInterfaceModel | SupportUserInitiativeModel | SupportSystemInitiativeModel, Field(discriminator='tactic_category')]

class MeteringModel(BaseModel):
    quality_attribute: Literal["Energy Efficiency"]
    tactic_category: Literal["Resource Monitoring"]
    tactic: Literal["Metering"]
    response: str


class StaticClassificationModel(BaseModel):
    quality_attribute: Literal["Energy Efficiency"]
    tactic_category: Literal["Resource Monitoring"]
    tactic: Literal["Static Classification"]
    response: str


class DynamicClassificationModel(BaseModel):
    quality_attribute: Literal["Energy Efficiency"]
    tactic_category: Literal["Resource Monitoring"]
    tactic: Literal["Dynamic Classification"]
    response: str


ResourceMonitoringModel = Annotated[MeteringModel | StaticClassificationModel | DynamicClassificationModel, Field(discriminator='tactic')]

class VerticalScalingModel(BaseModel):
    quality_attribute: Literal["Energy Efficiency"]
    tactic_category: Literal["Resource Allocation"]
    tactic: Literal["Vertical Scaling"]
    response: str


class HorizontalScalingModel(BaseModel):
    quality_attribute: Literal["Energy Efficiency"]
    tactic_category: Literal["Resource Allocation"]
    tactic: Literal["Horizontal Scaling"]
    response: str


class SchedulingModel(BaseModel):
    quality_attribute: Literal["Energy Efficiency"]
    tactic_category: Literal["Resource Allocation"]
    tactic: Literal["Scheduling"]
    response: str


class BrokeringModel(BaseModel):
    quality_attribute: Literal["Energy Efficiency"]
    tactic_category: Literal["Resource Allocation"]
    tactic: Literal["Brokering"]
    response: str


ResourceAllocationModel = Annotated[VerticalScalingModel | HorizontalScalingModel | SchedulingModel | BrokeringModel, Field(discriminator='tactic')]

class ServiceAdaptationModel(BaseModel):
    quality_attribute: Literal["Energy Efficiency"]
    tactic_category: Literal["Resource Adaptation"]
    tactic: Literal["Service Adaptation"]
    response: str


class IncreaseEfficiencyModel(BaseModel):
    quality_attribute: Literal["Energy Efficiency"]
    tactic_category: Literal["Resource Adaptation"]
    tactic: Literal["Increase Efficiency"]
    response: str


class ReduceOverheadModel(BaseModel):
    quality_attribute: Literal["Energy Efficiency"]
    tactic_category: Literal["Resource Adaptation"]
    tactic: Literal["Reduce Overhead"]
    response: str


ResourceAdaptationModel = Annotated[ServiceAdaptationModel | IncreaseEfficiencyModel | ReduceOverheadModel, Field(discriminator='tactic')]

EnergyEfficiencyModel = Annotated[ResourceMonitoringModel | ResourceAllocationModel | ResourceAdaptationModel, Field(discriminator='tactic_category')]

class TacticModel(BaseModel):
    architecture_tactic: Annotated[AvailabilityModel | InteroperabilityModel | ModifiabilityModel | PerformanceModel | SecurityModel | TestabilityModel | UsabilityModel | EnergyEfficiencyModel, Field(discriminator='quality_attribute')]
