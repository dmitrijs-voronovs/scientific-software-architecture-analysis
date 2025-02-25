from pydantic import BaseModel, Field
from typing import Literal, Annotated


class PingEchoModel(BaseModel):
    quality_attribute: Literal["Availability"]
    tactic_category: Literal["Detect Faults"]
    tactic: Literal["Ping / Echo"]
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

class TacticModel(BaseModel):
    architecture_tactic: Annotated[AvailabilityModel | PerformanceModel, Field(discriminator='quality_attribute')]
