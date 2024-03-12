/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using UnityEngine;
using UnityEditor;

namespace Oculus.Interaction.Editor
{
    /// <summary>
    /// Adds an [Optional] label in the inspector over any SerializedField with this attribute.
    /// </summary>
    [CustomPropertyDrawer(typeof(OptionalAttribute))]
    public class OptionalDrawer : DecoratorDrawer
    {
        private static readonly float HEADER_SIZE_AS_PERCENT = 0.25f;

        public override float GetHeight()
        {
            return base.GetHeight() * ( 1f + HEADER_SIZE_AS_PERCENT );
        }

        public override void OnGUI(Rect position)
        {
            position.y += GetHeight() * HEADER_SIZE_AS_PERCENT / ( 1f + HEADER_SIZE_AS_PERCENT );
            EditorGUI.LabelField(position, "[Optional]");
        }
    }
}
