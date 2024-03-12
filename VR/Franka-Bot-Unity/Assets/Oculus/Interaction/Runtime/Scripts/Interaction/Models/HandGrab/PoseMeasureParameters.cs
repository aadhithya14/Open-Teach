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

namespace Oculus.Interaction.HandGrab
{
    [Serializable]
    public struct PoseMeasureParameters
    {
        [SerializeField]
        [Min(0f)]
        private float _maxDistance;

        [SerializeField]
        [Range(0f, 1f)]
        private float _positionRotationWeight;

        public float MaxDistance => _maxDistance;
        public float PositionRotationWeight => _positionRotationWeight;

        public PoseMeasureParameters(float maxDistance, float positionRotationWeight)
        {
            _maxDistance = maxDistance;
            _positionRotationWeight = positionRotationWeight;
        }
    }
}
