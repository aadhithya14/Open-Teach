/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;
using UnityEngine;

namespace Oculus.Interaction.Input
{
    [Serializable]
    public class ControllerDataAsset : ICopyFrom<ControllerDataAsset>
    {
        public bool IsDataValid;
        public bool IsConnected;
        public bool IsTracked;
        public ControllerButtonUsage ButtonUsageMask;
        public Pose RootPose;
        public PoseOrigin RootPoseOrigin;
        public Pose PointerPose;
        public PoseOrigin PointerPoseOrigin;
        public ControllerDataSourceConfig Config;

            public void CopyFrom(ControllerDataAsset source)
        {
            IsDataValid = source.IsDataValid;
            IsConnected = source.IsConnected;
            IsTracked = source.IsTracked;
            Config = source.Config;
            CopyPosesAndStateFrom(source);
        }

        public void CopyPosesAndStateFrom(ControllerDataAsset source)
        {
            ButtonUsageMask = source.ButtonUsageMask;
            RootPose = source.RootPose;
            RootPoseOrigin = source.RootPoseOrigin;
            PointerPose = source.PointerPose;
            PointerPoseOrigin = source.PointerPoseOrigin;
        }
    }
}
