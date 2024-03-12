/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System.Collections.Generic;
using UnityEditor;

namespace Oculus.Interaction.PoseDetection.Editor
{
    [CustomEditor(typeof(TransformFeatureStateThresholds))]
    class TransformFeatureStateThresholdModel
        : FeatureStateThresholdsEditor<TransformFeature>
    {
        protected override IReadOnlyDictionary<TransformFeature, FeatureDescription> CreateFeatureDescriptions()
        {
            return TransformFeatureProperties.FeatureDescriptions;
        }
    }
}
