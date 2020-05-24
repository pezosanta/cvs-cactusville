import glob
import cv2 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

TrainFULLPath                   = '../../logs/cvs-cactusville/trafficSignsDataset/trainFULL/'
TestFULLPath                    = '../../logs/cvs-cactusville/trafficSignsDataset/testFULL/'
JPG                             = '*.jpg'

# Train image folder paths
BumpTrainPath                   = glob.glob(TrainFULLPath + 'Bump/' + JPG)
BumpyRoadTrainPath              = glob.glob(TrainFULLPath + 'Bumpy road/' + JPG)
BusStopTrainPath                = glob.glob(TrainFULLPath + 'Bus stop/' + JPG)
ChildrenTrainPath               = glob.glob(TrainFULLPath + 'Children/' + JPG)
CrossingBlueTrainPath           = glob.glob(TrainFULLPath + 'Crossing (blue)/' + JPG)
CrossingRedTrainPath            = glob.glob(TrainFULLPath + 'Crossing (red)/' + JPG)
CyclistsTrainPath               = glob.glob(TrainFULLPath + 'Cyclists/' + JPG)
DangerTrainPath                 = glob.glob(TrainFULLPath + 'Danger (other)/' + JPG)
DangerousLeftTurnTrainPath      = glob.glob(TrainFULLPath + 'Dangerous left turn/' + JPG)
DangerousRightTurnTrainPath     = glob.glob(TrainFULLPath + 'Dangerous right turn/' + JPG)
GiveWayTrainPath                = glob.glob(TrainFULLPath + 'Give way/' + JPG)
GoAheadTrainPath                = glob.glob(TrainFULLPath + 'Go ahead/' + JPG)
GoAheadOrLeftTrainPath          = glob.glob(TrainFULLPath + 'Go ahead or left/' + JPG)
GoAheadOrRightTrainPath         = glob.glob(TrainFULLPath + 'Go ahead or right/' + JPG)
GoAroundEitherWayTrainPath      = glob.glob(TrainFULLPath + 'Go around either way/' + JPG)
GoAroundLeftTrainPath           = glob.glob(TrainFULLPath + 'Go around left/' + JPG)
GoAroundRightTrainPath          = glob.glob(TrainFULLPath + 'Go around right/' + JPG)
IntersectionTrainPath           = glob.glob(TrainFULLPath + 'Intersection/' + JPG)
Limit100TrainPath               = glob.glob(TrainFULLPath + 'Limit 100/' + JPG)
Limit120TrainPath               = glob.glob(TrainFULLPath + 'Limit 120/' + JPG)
Limit20TrainPath                = glob.glob(TrainFULLPath + 'Limit 20/' + JPG)
Limit30TrainPath                = glob.glob(TrainFULLPath + 'Limit 30/' + JPG)
Limit50TrainPath                = glob.glob(TrainFULLPath + 'Limit 50/' + JPG)
Limit60TrainPath                = glob.glob(TrainFULLPath + 'Limit 60/' + JPG)
Limit70TrainPath                = glob.glob(TrainFULLPath + 'Limit 70/' + JPG)
Limit80TrainPath                = glob.glob(TrainFULLPath + 'Limit 80/' + JPG)
Limit80OverTrainPath            = glob.glob(TrainFULLPath + 'Limit 80 over/' + JPG)
LimitOverTrainPath              = glob.glob(TrainFULLPath + 'Limit over/' + JPG)
MainRoadTrainPath               = glob.glob(TrainFULLPath + 'Main road/' + JPG)
MainRoadOverTrainPath           = glob.glob(TrainFULLPath + 'Main road over/' + JPG)
MultipleDangerousTurnsTrainPath = glob.glob(TrainFULLPath + 'Multiple dangerous turns/' + JPG)
NarrowRoadLeftTrainPath         = glob.glob(TrainFULLPath + 'Narrow road (left)/' + JPG)
NarrowRoadRightTrainPath        = glob.glob(TrainFULLPath + 'Narrow road (right)/' + JPG)
NoEntryTrainPath                = glob.glob(TrainFULLPath + 'No entry/' + JPG)
NoEntryBothDirectionsTrainPath  = glob.glob(TrainFULLPath + 'No entry (both directions)/' + JPG)
NoEntryTruckTrainPath           = glob.glob(TrainFULLPath + 'No entry (truck)/' + JPG)
NoStoppingTrainPath             = glob.glob(TrainFULLPath + 'No stopping/' + JPG)
NoTakeoverTrainPath             = glob.glob(TrainFULLPath + 'No takeover/' + JPG)
NoTakeoverTruckTrainPath        = glob.glob(TrainFULLPath + 'No takeover (truck)/' + JPG)
NoTakeoverTruckEndTrainPath     = glob.glob(TrainFULLPath + 'No takeover (truck) end/' + JPG)
NoTakeoverEndTrainPath          = glob.glob(TrainFULLPath + 'No takeover end/' + JPG)
NoWaitingTrainPath              = glob.glob(TrainFULLPath + 'No waiting/' + JPG)
OneWayRoadTrainPath             = glob.glob(TrainFULLPath + 'One way road/' + JPG)
ParkingTrainPath                = glob.glob(TrainFULLPath + 'Parking/' + JPG)
RoadWorksTrainPath              = glob.glob(TrainFULLPath + 'Road works/' + JPG)
RoundAboutTrainPath             = glob.glob(TrainFULLPath + 'Roundabout/' + JPG)
SlipperyRoadTrainPath           = glob.glob(TrainFULLPath + 'Slippery road/' + JPG)
StopTrainPath                   = glob.glob(TrainFULLPath + 'Stop/' + JPG)
TrafficLightTrainPath           = glob.glob(TrainFULLPath + 'Traffic light/' + JPG)
TrainCrossingTrainPath          = glob.glob(TrainFULLPath + 'Train crossing/' + JPG)
TrainCrossingNoBarrierTrainPath = glob.glob(TrainFULLPath + 'Train crossing (no barrier)/' + JPG)
WildAnimalsTrainPath            = glob.glob(TrainFULLPath + 'Wild animals/' + JPG)
XPriorityTrainPath              = glob.glob(TrainFULLPath + 'X - Priority/' + JPG)
XTurnLeftTrainPath              = glob.glob(TrainFULLPath + 'X - Turn left/' + JPG)
XTurnRightTrainPath             = glob.glob(TrainFULLPath + 'X - Turn right/' + JPG)

# Test image folder paths
BumpTestPath                    = glob.glob(TestFULLPath + 'Bump/' + JPG)
BumpyRoadTestPath               = glob.glob(TestFULLPath + 'Bumpy road/'+ JPG)
BusStopTestPath                 = glob.glob(TestFULLPath + 'Bus stop/'+ JPG)
ChildrenTestPath                = glob.glob(TestFULLPath + 'Children/'+ JPG)
CrossingBlueTestPath            = glob.glob(TestFULLPath + 'Crossing (blue)/'+ JPG)
CrossingRedTestPath             = glob.glob(TestFULLPath + 'Crossing (red)/'+ JPG)
CyclistsTestPath                = glob.glob(TestFULLPath + 'Cyclists/'+ JPG)
DangerTestPath                  = glob.glob(TestFULLPath + 'Danger (other)/'+ JPG)
DangerousLeftTurnTestPath       = glob.glob(TestFULLPath + 'Dangerous left turn/'+ JPG)
DangerousRightTurnTestPath      = glob.glob(TestFULLPath + 'Dangerous right turn/'+ JPG)
GiveWayTestPath                 = glob.glob(TestFULLPath + 'Give way/'+ JPG)
GoAheadTestPath                 = glob.glob(TestFULLPath + 'Go ahead/'+ JPG)
GoAheadOrLeftTestPath           = glob.glob(TestFULLPath + 'Go ahead or left/'+ JPG)
GoAheadOrRightTestPath          = glob.glob(TestFULLPath + 'Go ahead or right/'+ JPG)
GoAroundEitherWayTestPath       = glob.glob(TestFULLPath + 'Go around either way/'+ JPG)
GoAroundLeftTestPath            = glob.glob(TestFULLPath + 'Go around left/'+ JPG)
GoAroundRightTestPath           = glob.glob(TestFULLPath + 'Go around right/'+ JPG)
IntersectionTestPath            = glob.glob(TestFULLPath + 'Intersection/'+ JPG)
Limit100TestPath                = glob.glob(TestFULLPath + 'Limit 100/'+ JPG)
Limit120TestPath                = glob.glob(TestFULLPath + 'Limit 120/'+ JPG)
Limit20TestPath                 = glob.glob(TestFULLPath + 'Limit 20/'+ JPG)
Limit30TestPath                 = glob.glob(TestFULLPath + 'Limit 30/'+ JPG)
Limit50TestPath                 = glob.glob(TestFULLPath + 'Limit 50/'+ JPG)
Limit60TestPath                 = glob.glob(TestFULLPath + 'Limit 60/'+ JPG)
Limit70TestPath                 = glob.glob(TestFULLPath + 'Limit 70/'+ JPG)
Limit80TestPath                 = glob.glob(TestFULLPath + 'Limit 80/'+ JPG)
Limit80OverTestPath             = glob.glob(TestFULLPath + 'Limit 80 over/'+ JPG)
LimitOverTestPath               = glob.glob(TestFULLPath + 'Limit over/'+ JPG)
MainRoadTestPath                = glob.glob(TestFULLPath + 'Main road/'+ JPG)
MainRoadOverTestPath            = glob.glob(TestFULLPath + 'Main road over/'+ JPG)
MultipleDangerousTurnsTestPath  = glob.glob(TestFULLPath + 'Multiple dangerous turns/'+ JPG)
NarrowRoadLeftTestPath          = glob.glob(TestFULLPath + 'Narrow road (left)/'+ JPG)
NarrowRoadRightTestPath         = glob.glob(TestFULLPath + 'Narrow road (right)/'+ JPG)
NoEntryTestPath                 = glob.glob(TestFULLPath + 'No entry/'+ JPG)
NoEntryBothDirectionsTestPath   = glob.glob(TestFULLPath + 'No entry (both directions)/'+ JPG)
NoEntryTruckTestPath            = glob.glob(TestFULLPath + 'No entry (truck)/'+ JPG)
NoStoppingTestPath              = glob.glob(TestFULLPath + 'No stopping/'+ JPG)
NoTakeoverTestPath              = glob.glob(TestFULLPath + 'No takeover/'+ JPG)
NoTakeoverTruckTestPath         = glob.glob(TestFULLPath + 'No takeover (truck)/'+ JPG)
NoTakeoverTruckEndTestPath      = glob.glob(TestFULLPath + 'No takeover (truck) end/'+ JPG)
NoTakeoverEndTestPath           = glob.glob(TestFULLPath + 'No takeover end/'+ JPG)
NoWaitingTestPath               = glob.glob(TestFULLPath + 'No waiting/'+ JPG)
OneWayRoadTestPath              = glob.glob(TestFULLPath + 'One way road/'+ JPG)
ParkingTestPath                 = glob.glob(TestFULLPath + 'Parking/'+ JPG)
RoadWorksTestPath               = glob.glob(TestFULLPath + 'Road works/'+ JPG)
RoundAboutTestPath              = glob.glob(TestFULLPath + 'Roundabout/'+ JPG)
SlipperyRoadTestPath            = glob.glob(TestFULLPath + 'Slippery road/'+ JPG)
StopTestPath                    = glob.glob(TestFULLPath + 'Stop/'+ JPG)
TrafficLightTestPath            = glob.glob(TestFULLPath + 'Traffic light/'+ JPG)
TrainCrossingTestPath           = glob.glob(TestFULLPath + 'Train crossing/'+ JPG)
TrainCrossingNoBarrierTestPath  = glob.glob(TestFULLPath + 'Train crossing (no barrier)/'+ JPG)
WildAnimalsTestPath             = glob.glob(TestFULLPath + 'Wild animals/'+ JPG)
XPriorityTestPath               = glob.glob(TestFULLPath + 'X - Priority/'+ JPG)
XTurnLeftTestPath               = glob.glob(TestFULLPath + 'X - Turn left/'+ JPG)
XTurnRightTestPath              = glob.glob(TestFULLPath + 'X - Turn right/'+ JPG)

def createFullPathLists(mode):
    if mode == 'Train' or mode == 'Val':
        TrainValFULLPathList = [
            BumpTrainPath, BumpyRoadTrainPath, BusStopTrainPath, ChildrenTrainPath, CrossingBlueTrainPath, CrossingRedTrainPath, CyclistsTrainPath, DangerTrainPath,
            DangerousLeftTurnTrainPath, DangerousRightTurnTrainPath, GiveWayTrainPath, GoAheadTrainPath, GoAheadOrLeftTrainPath, GoAheadOrRightTrainPath,
            GoAroundEitherWayTrainPath, GoAroundLeftTrainPath, GoAroundRightTrainPath, IntersectionTrainPath, Limit100TrainPath, Limit120TrainPath, Limit20TrainPath,
            Limit30TrainPath, Limit50TrainPath, Limit60TrainPath, Limit70TrainPath, Limit80TrainPath, Limit80OverTrainPath, LimitOverTrainPath, MainRoadTrainPath,
            MainRoadOverTrainPath, MultipleDangerousTurnsTrainPath, NarrowRoadLeftTrainPath, NarrowRoadRightTrainPath, NoEntryTrainPath, NoEntryBothDirectionsTrainPath,
            NoEntryTruckTrainPath, NoStoppingTrainPath, NoTakeoverTrainPath, NoTakeoverTruckTrainPath, NoTakeoverTruckEndTrainPath, NoTakeoverEndTrainPath,
            NoWaitingTrainPath, OneWayRoadTrainPath, ParkingTrainPath, RoadWorksTrainPath, RoundAboutTrainPath, SlipperyRoadTrainPath, StopTrainPath, TrafficLightTrainPath,
            TrainCrossingTrainPath, TrainCrossingNoBarrierTrainPath, WildAnimalsTrainPath, XPriorityTrainPath, XTurnLeftTrainPath, XTurnRightTrainPath]

        # Splittin the TrainVal dataset to Train and Val datasets
        LenTrainValPaths    = len(TrainValFULLPathList[0])
        train_factor        = 0.95                              # 2000*0.95 = 1900 ==> split 2000 to 1900 training and 100 validation images
        TrainFULLPathList   = []
        ValFULLPathList     = []

        for i in range(len(TrainValFULLPathList)):
            TrainFULLPathList.append(TrainValFULLPathList[i][:int(LenTrainValPaths*train_factor)])
            ValFULLPathList.append(TrainValFULLPathList[i][int(LenTrainValPaths*train_factor):])
        
        if mode == 'Train':
            return TrainFULLPathList
        elif mode == 'Val':
            return ValFULLPathList
        
    elif mode == 'Test':
        TestFULLPathList = [
            BumpTestPath, BumpyRoadTestPath, BusStopTestPath, ChildrenTestPath, CrossingBlueTestPath, CrossingRedTestPath, CyclistsTestPath, DangerTestPath,
            DangerousLeftTurnTestPath, DangerousRightTurnTestPath, GiveWayTestPath, GoAheadTestPath, GoAheadOrLeftTestPath, GoAheadOrRightTestPath,
            GoAroundEitherWayTestPath, GoAroundLeftTestPath, GoAroundRightTestPath, IntersectionTestPath, Limit100TestPath, Limit120TestPath, Limit20TestPath,
            Limit30TestPath, Limit50TestPath, Limit60TestPath, Limit70TestPath, Limit80TestPath, Limit80OverTestPath, LimitOverTestPath, MainRoadTestPath,
            MainRoadOverTestPath, MultipleDangerousTurnsTestPath, NarrowRoadLeftTestPath, NarrowRoadRightTestPath, NoEntryTestPath, NoEntryBothDirectionsTestPath,
            NoEntryTruckTestPath, NoStoppingTestPath, NoTakeoverTestPath, NoTakeoverTruckTestPath, NoTakeoverTruckEndTestPath, NoTakeoverEndTestPath,
            NoWaitingTestPath, OneWayRoadTestPath, ParkingTestPath, RoadWorksTestPath, RoundAboutTestPath, SlipperyRoadTestPath, StopTestPath, TrafficLightTestPath,
            TrainCrossingTestPath, TrainCrossingNoBarrierTestPath, WildAnimalsTestPath, XPriorityTestPath, XTurnLeftTestPath, XTurnRightTestPath]
    
        return TestFULLPathList

def createOneHotAnnotations(FULLPathList):
    FULLAnnotationList = [0] * len(FULLPathList)            # Creating a list with 55 elements of zeros
    for i in range(len(FULLAnnotationList)):
        FULLAnnotationList[i] = [0] * len(FULLPathList[i])
        for j in range(len(FULLAnnotationList[i])):
            FULLAnnotationList[i][j] = [0] * len(FULLPathList)
            FULLAnnotationList[i][j][i] = 1
    return FULLAnnotationList

def createFlatList(stagedList):
    flatList                = [item for category in stagedList for item in category]
    return flatList

def createAllFlatLists(mode):
    FULLPathList            = createFullPathLists(mode)
    FULLAnnotationList      = createOneHotAnnotations(FULLPathList)

    FlatPathList            = createFlatList(FULLPathList)
    FlatAnnotationList      = createFlatList(FULLAnnotationList)
    
    return FlatPathList, FlatAnnotationList

def shuffleFlatLists(FlatPathList, FlatAnnotationList):
    FlatPathArray           = np.array(FlatPathList)
    FlatAnnotationArray     = np.array(FlatAnnotationList)

    indices                 = np.arange(FlatPathArray.shape[0])
    np.random.shuffle(indices)

    FlatPathArray           = FlatPathArray[indices]
    FlatAnnotationArray     = FlatAnnotationArray[indices]

    return FlatPathArray, FlatAnnotationArray

class TSDataset(Dataset):
    def __init__(self, mode):
        self.device         = torch.device('cuda:0')
        self.mode           = mode
        self.FlatPathList, self.FlatAnnotationList  = createAllFlatLists(self.mode)

    def __getitem__(self, index):
        FlatPathArray, FlatAnnotationArray          = shuffleFlatLists(self.FlatPathList, self.FlatAnnotationList)
        
        numpyImage          = cv2.imread(FlatPathArray[index])
        numpyImage          = cv2.cvtColor(numpyImage, cv2.COLOR_BGR2RGB)

        numpyAnnotation     = FlatAnnotationArray[index]

        tensorImage         = torch.from_numpy(numpyImage / 255.0).float().to(device = self.device).permute(2, 0, 1)
        tensorAnnotation    = torch.from_numpy(numpyAnnotation).float().to(device = self.device)

        return tensorImage, tensorAnnotation#, FlatPathArray[index]

    def __len__(self): 
      return len(self.FlatPathList)