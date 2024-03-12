/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

namespace Oculus.Interaction.PoseDetection
{
    public class FeatureStateDescription
    {
        public FeatureStateDescription(string id, string name)
        {
            Id = id;
            Name = name;
        }

        public string Id { get; }
        public string Name { get; }
    }

    public class FeatureDescription
    {
        public FeatureDescription(string shortDescription, string description,
            float minValueHint, float maxValueHint,
            FeatureStateDescription[] featureStates)
        {
            ShortDescription = shortDescription;
            Description = description;
            MinValueHint = minValueHint;
            MaxValueHint = maxValueHint;
            FeatureStates = featureStates;
        }

        public string ShortDescription { get; }
        public string Description { get; }
        public float MinValueHint { get; }
        public float MaxValueHint { get; }

        /// <summary>
        /// A hint to the editor on which feature states to provide by default, and in which order
        /// they should appear.
        /// The underlying system will accept other ranges; this is just for the UI.
        /// </summary>
        public FeatureStateDescription[] FeatureStates { get; }
    }
}
