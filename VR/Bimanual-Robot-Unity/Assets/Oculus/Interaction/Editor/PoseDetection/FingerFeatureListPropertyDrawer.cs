/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.PoseDetection.Editor.Model;
using System;
using System.Linq;
using UnityEditor;

namespace Oculus.Interaction.PoseDetection.Editor
{
    [CustomPropertyDrawer(typeof(ShapeRecognizer.FingerFeatureConfigList))]
    public class FingerFeatureListPropertyDrawer : FeatureListPropertyDrawer
    {
        [Flags]
        enum FingerFeatureFlags
        {
            Curl = 1 << 0,
            Flexion = 1 << 1,
            Abduction = 1 << 2,
            Opposition = 1 << 3
        }

        protected override Enum FlagsToEnum(uint flags)
        {
            return (FingerFeatureFlags)flags;
        }

        protected override uint EnumToFlags(Enum flags)
        {
            return (uint)(FingerFeatureFlags)flags;
        }

        protected override string FeatureToString(int featureIdx)
        {
            return ((FingerFeature)featureIdx).ToString();
        }

        protected override FeatureStateDescription[] GetStatesForFeature(int featureIdx)
        {
            return FingerFeatureProperties.FeatureDescriptions[(FingerFeature)featureIdx].FeatureStates;
        }

        protected override FeatureConfigList CreateModel(SerializedProperty property)
        {
            var descriptions = FingerFeatureProperties.FeatureDescriptions
                .ToDictionary(p => (int)p.Key, p => p.Value);

            return new FeatureConfigList(property.FindPropertyRelative("_value"),
                descriptions);
        }
    }
}
