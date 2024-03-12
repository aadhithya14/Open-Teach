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
namespace Oculus.Interaction
{
    public class ConditionalHideAttribute : PropertyAttribute
    {
        public string ConditionalFieldPath { get; private set; }
        public object HideValue { get; private set; }


        public ConditionalHideAttribute(string fieldName, object hideValue)
        {
            ConditionalFieldPath = fieldName;
            HideValue = hideValue;
        }
    }
}
