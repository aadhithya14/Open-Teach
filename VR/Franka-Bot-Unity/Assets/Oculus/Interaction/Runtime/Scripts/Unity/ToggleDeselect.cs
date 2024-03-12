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
using UnityEngine.UI;
using UnityEngine.EventSystems;

namespace Oculus.Interaction
{
    /// <summary>
    /// Override Toggle to clear state on drag while still bubbling events up through
    /// the hierarchy. Particularly useful for buttons inside of scroll views.
    /// </summary>
    public class ToggleDeselect : Toggle
    {
        [SerializeField]
        private bool _clearStateOnDrag = false;

        public bool ClearStateOnDrag
        {
            get
            {
                return _clearStateOnDrag;
            }

            set

            {
                _clearStateOnDrag = value;
            }
        }

        public void OnBeginDrag(PointerEventData pointerEventData)
        {
            if (!_clearStateOnDrag)
            {
                return;
            }
            InstantClearState();
            DoStateTransition(SelectionState.Normal, true);
            ExecuteEvents.ExecuteHierarchy(
                transform.parent.gameObject,
                pointerEventData,
                ExecuteEvents.beginDragHandler
            );
        }
    }
}
