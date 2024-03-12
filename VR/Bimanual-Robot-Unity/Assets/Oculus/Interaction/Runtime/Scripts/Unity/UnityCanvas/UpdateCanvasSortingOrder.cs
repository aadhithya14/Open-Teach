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

namespace Oculus.Interaction.UnityCanvas
{
    /// <summary>
    /// Dropdowns menus in Unity are automatically set to sorting order 30000, which
    /// does not play nicely with world space UIs.
    /// Meant to be used in conjunction with an EventTrigger on a given Dropdown, this component
    /// can be used to set a different sorting order on this and any child canvas.
    /// </summary>
    public class UpdateCanvasSortingOrder : MonoBehaviour
    {
        public void SetCanvasSortingOrder(int sortingOrder)
        {
            Canvas[] canvases = transform.parent.gameObject.GetComponentsInChildren<Canvas>();
            if (canvases == null) return;
            foreach (Canvas canvas in canvases)
            {
                canvas.sortingOrder = sortingOrder;
            }
        }
    }
}
